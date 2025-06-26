from EfficientNetFoodModel import FoodClassifierEfficientNetB3Based
from ResnetFoodModel import FoodClassifierResNetBased
from recipe_utils import process_dish
from PIL import Image
from torchvision import transforms
import streamlit as st
import torch

NUM_CLASSES = 103
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT1 = 0.6
WEIGHT2 = 0.4

@st.cache_resource
def load_models():
    model1 = FoodClassifierResNetBased(num_classes=NUM_CLASSES).to(DEVICE)
    model2 = FoodClassifierEfficientNetB3Based(num_classes=NUM_CLASSES).to(DEVICE)
    state1 = torch.load("resnet_model.pth", map_location=DEVICE)
    state2 = torch.load("efficientnet_model.pth", map_location=DEVICE)
    model1.load_state_dict(state1["model1_state_dict_resnet"])
    model2.load_state_dict(state2["model2_state_dict_efficientnet"])
    class_names = state1["class_names"]

    return model1, model2, class_names

@st.cache_resource
def get_tta_transforms():
    return [
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    ]


model1, model2, class_names = load_models()

st.title("Food Recognition and Recipe Finder With Nutrition Info")
st.write("Upload an image of food to get its name, recipe, and nutritional information.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    rgba_image = image.convert("RGBA")
    st.image(rgba_image, caption="Uploaded Image", use_container_width=True)

    tta_transforms = get_tta_transforms()

    # Apply TTA transforms and get predictions
    model1.eval()
    model2.eval()
    with torch.inference_mode():
        outputs = []
        for transform in tta_transforms:
            transformed_image = transform(image).unsqueeze(0).to(DEVICE)
            output1 = model1(transformed_image)
            output2 = model2(transformed_image)
            combined_output = WEIGHT1 * output1 + WEIGHT2 * output2
            outputs.append(combined_output)

    # Average the outputs
    final_output = sum(outputs) / len(outputs)
    predicted_class = final_output.argmax(dim=1).item()

    dish_name = class_names[predicted_class]
    st.write(f"Predicted Dish: {dish_name}".replace("_", " ").title())
    
    process_dish(dish_name)
    
else:
    st.write("Please upload an image to get started.")
