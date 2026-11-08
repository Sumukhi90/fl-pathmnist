
import torch
import torch.nn as nn


def train_local_model(
    model,
    train_loader,
    device="cpu",
    epochs=1,
    learning_rate=0.01,
    mu=0.0
):
    """
    Train one client's local model.

    mu = 0.0: FedAvg
    mu > 0.0: FedProx

    Returns:
        Updated model state dictionary
        Average classification loss
        Local training accuracy
    """

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    if mu < 0:
        raise ValueError("mu cannot be negative.")

    if len(train_loader.dataset) == 0:
        raise ValueError("Client dataset cannot be empty.")

    model = model.to(device)
    model.train()

    # Preserve the initial global parameters.
    global_params = [
        param.detach().clone()
        for param in model.parameters()
    ]

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device).view(-1)

            optimizer.zero_grad()

            outputs = model(images)

            # Classification loss.
            classification_loss = criterion(
                outputs, labels
            )

            # FedProx regularization.
            objective_loss = classification_loss

            if mu > 0:

                proximal_term = torch.zeros(
                    (), device=device
                )

                for param, global_param in zip(
                    model.parameters(),
                    global_params
                ):
                    proximal_term += torch.sum(
                        (param - global_param) ** 2
                    )

                objective_loss = (
                    classification_loss
                    + (mu / 2.0) * proximal_term
                )

            # Optimize the complete objective.
            objective_loss.backward()
            optimizer.step()

            # Report classification loss only.
            total_loss += (
                classification_loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return (
        model.state_dict(),
        average_loss,
        accuracy
    )
