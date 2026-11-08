
import torch
import torch.nn as nn


def evaluate_model(model, test_loader, device="cpu"):
    """
    Evaluate the global model on an independent test dataset.

    Computes:
        - Average cross-entropy test loss
        - Classification accuracy

    Args:
        model: Global PyTorch model.
        test_loader: DataLoader containing the test dataset.
        device: CPU or CUDA device.

    Returns:
        Dictionary containing test loss and test accuracy.

    Raises:
        ValueError: If the test dataset is empty.
    """

    # --------------------------------------------------------
    # 1. VALIDATE TEST DATASET
    # --------------------------------------------------------

    if len(test_loader.dataset) == 0:
        raise ValueError("Test dataset cannot be empty.")

    # --------------------------------------------------------
    # 2. SET MODEL TO EVALUATION MODE
    # --------------------------------------------------------

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # --------------------------------------------------------
    # 3. EVALUATE WITHOUT GRADIENT COMPUTATION
    # --------------------------------------------------------

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device).view(-1)

            outputs = model(images)

            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += (
                loss.item() * labels.size(0)
            )

            total_correct += (
                (predictions == labels).sum().item()
            )

            total_samples += labels.size(0)

    # --------------------------------------------------------
    # 4. CALCULATE FINAL METRICS
    # --------------------------------------------------------

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    # --------------------------------------------------------
    # 5. RETURN EVALUATION RESULTS
    # --------------------------------------------------------

    return {
        "test_loss": average_loss,
        "test_accuracy": accuracy
    }
