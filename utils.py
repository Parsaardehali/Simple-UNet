import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_results(model, dataset, num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        input_img, target_img = dataset[i]
        input_img = input_img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_img)
        
        input_img = input_img.cpu().squeeze().numpy()
        target_img = target_img.squeeze().numpy()
        output = output.cpu().squeeze().numpy()
        
        axes[i, 0].imshow(input_img, cmap='gray')
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(target_img, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(output > 0.5, cmap='gray')  # Threshold at 0.5
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_dice_score(model, dataset, device, threshold=0.5):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for i, (input_img, target_img) in enumerate(dataset):
            
            # Remove the unsqueeze operation
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            
            try:
                output = model(input_img)
                
                predicted = (output > threshold).float()
                
                intersection = (predicted * target_img).sum(dim=(1, 2, 3))
                union = predicted.sum(dim=(1, 2, 3)) + target_img.sum(dim=(1, 2, 3))
                
                dice = (2.0 * intersection) / (union + 1e-7)
                dice_scores.extend(dice.cpu().tolist())
            except RuntimeError as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

    if dice_scores:
        average_dice = np.mean(dice_scores)
        return average_dice
    else:
        print("No valid samples processed.")
        return None