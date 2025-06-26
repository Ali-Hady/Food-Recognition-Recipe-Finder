import requests
import streamlit as st

def get_recipe_by_name(dish_name):
    url = f"https://www.themealdb.com/api/json/v1/1/search.php?s={dish_name}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        meals = data.get("meals")

        if meals:
            st.header("Recipe Found!")
            meal = meals[0]  # Take the first match
            st.write("🍽️ Suggested Meal:", meal["strMeal"])
            st.write("🌍 Area:", meal["strArea"])
            st.write("📖 Instructions:", meal["strInstructions"])
            st.write("📹 YouTube:", meal["strYoutube"])
        else:
            st.write("No recipe found for:", dish_name)
        
        return meals
    
    st.write("Error fetching data:", response.status_code)
    st.write("Please try again later.")

    return None

def get_nutrient_by_food(label):
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {"search_terms": label, "search_simple": 1, "json": 1}
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        return None
    
    data = resp.json()
    if data["products"]:
        prod = data["products"][0]
        nutr = prod.get("nutriments", {})
        return {
            "energy_kcal per 100g": nutr.get("energy-kcal_100g"),
            "fat per 100g": nutr.get("fat_100g"),
            "carbs per 100g": nutr.get("carbohydrates_100g"),
            "protein per 100g": nutr.get("proteins_100g"),
        }
        
    return None

def process_dish(dish_name):
    meals = get_recipe_by_name(dish_name)
    
    meal = meals[0] if meals else None
    if meal:
        with st.spinner("🔎 Fetching nutritional info..."):
            ingredients = []
            energy = []
            fat = []
            carbs = []
            protein = []
            for i in range(1, 21):
                ingredient = meal.get(f"strIngredient{i}")
                measure = meal.get(f"strMeasure{i}")

                if ingredient and ingredient.strip():
                    nutrient_info = get_nutrient_by_food(ingredient)
                    if nutrient_info:
                        ingredients.append(f"{ingredient} ({measure}):")
                        energy.append(f"{nutrient_info['energy_kcal per 100g']} kcal")
                        fat.append(f"{nutrient_info['fat per 100g']} g")
                        carbs.append(f"{nutrient_info['carbs per 100g']} g")
                        protein.append(f"{nutrient_info['protein per 100g']} g")
                        
        st.header("Nutritional Information")            
        st.dataframe({
            "Ingredient": ingredients,
            "Energy (kcal/100g)": energy,
            "Fat (g/100g)": fat,
            "Carbs (g/100g)": carbs,
            "Protein (g/100g)": protein
        })        

if __name__ == "__main__":
    dish_name = st.text_input("Enter the name of the dish:")
    if dish_name:
        process_dish(dish_name)