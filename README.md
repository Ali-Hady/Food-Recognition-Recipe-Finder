# Food-Recognition-Recipe-Finder
🍽️ Foodify: a food recognition app that predicts the dish and fetches recipes and nutritional information.
Built with Computer Vision + API integration (for the recipes), it’s trained on the challenging Food101 dataset, known for its fine-grained categories and visually similar classes, plus a custom extension of Egyptian dishes.

Technical Highlights:
ResNet50 fine-tuned with ReduceLROnPlateau, reaching 88.5% validation accuracy
EfficientNetB3 with Cosine Annealing, reaching 87.47% validation accuracy
Final model is a weighted ensemble (Slightly favoring ResNet) + Test Time Augmentation, reaching 90.12% accuracy
This is close to state-of-the-art performance; most 90%+ Food101 benchmarks use much deeper architectures or additional metadata. That last 5% really pushed my understanding of training dynamics, regularization, and optimization!

After the model predicts the dish, the app automatically:
- Fetches the recipe from TheMealDB API (plus a video tutorial)
- Retrieves nutritional info from OpenFoodFacts, ingredient by ingredient
- Displays everything neatly using Streamlit, including a table of macros per item
It currently supports 103 classes, and I’m planning to keep expanding both the dataset and functionality.

Try it yourself: https://lnkd.in/d69eRBep
