import torch
import copy
from tqdm import tqdm

from src.cosine_annealing_with_warmup import CosineAnnealingWithWarmup

def train(
          model: torch.nn.Module,
          train_loader,
          val_loader,
          criterion: torch.nn.CrossEntropyLoss,
          optimizer,
          num_epochs: int = 10,
          warmup_epochs: int = 5,
          scaler=None,
          patience: int = 7,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_type = device.type
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
        amp = device_type == "cuda"

        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            num_epochs=num_epochs,
            warmup_epochs=warmup_epochs
        )

        model = model.to(device)

        best_val_loss = float("inf")
        best_model_weights = None
        epochs_no_improve = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            # ---- Training ----
            model.train()
            total_loss = 0.0
            n_corrects = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=amp):
                    logits = model(images)
                    loss = criterion(logits, labels)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                _, preds = torch.max(logits, 1)
                total_loss += loss.item() * images.size(0)
                n_corrects += (preds == labels).sum().item()

                pbar.set_postfix(loss=loss.item())

            scheduler.step()

            train_loss = total_loss / len(train_loader.dataset)
            train_acc = n_corrects / len(train_loader.dataset)

            # ---- Evaluation ----
            model.eval()
            total_loss = 0.0
            n_corrects = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=amp):
                        logits = model(images)
                        loss = criterion(logits, labels)

                    _, preds = torch.max(logits, 1)
                    total_loss += loss.item() * images.size(0)
                    n_corrects += (preds == labels).sum().item()

            val_loss = total_loss / len(val_loader.dataset)
            val_acc = n_corrects / len(val_loader.dataset)

            # Save history
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Early stopping based on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{num_epochs}] lr: {lr:.6f} | train loss: {train_loss:.4f}, train acc: {train_acc:.4f} | val loss: {val_loss:.4f}, val acc: {val_acc:.4f} | patience: {patience - epochs_no_improve}")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} — val loss didn't improve for {patience} epochs.")
                break

        # Restore best model
        if best_model_weights:
            model.load_state_dict(best_model_weights)
            print(f"Best val loss: {best_val_loss:.4f} — best weights restored.")

        return model, history