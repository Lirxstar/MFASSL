"""Full fine-tuning evaluation: encoder + linear head, all trainable."""

from .classifier import ClassifierModel, train_classifier, evaluate_classifier

def run_finetune(encoder, num_classes, train_loader, test_loader, epochs=1,
                 lr=1e-4, device="cpu", max_steps=None, flip_reduction="per_image",
                 save_predictions=None):

    model = ClassifierModel(encoder, num_classes, freeze_encoder=False)
    train_classifier(model, train_loader, epochs=epochs, lr=lr, device=device,
                     max_steps=max_steps)
    return evaluate_classifier(model, test_loader, device=device,
                               flip_reduction=flip_reduction,
                               save_predictions=save_predictions)
