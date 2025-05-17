from train_classifier import train_icon_classifier

train_data_path = "data" 
model_output_path = "models/resnet_icon.pt"

# Train the model
model, class_names = train_icon_classifier(
    data_dir=train_data_path,
    output_model=model_output_path,
    num_epochs=20,     # Can change
    batch_size=16
)