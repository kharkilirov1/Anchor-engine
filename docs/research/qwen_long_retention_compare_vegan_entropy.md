# Qwen Long Retention Compare

Date: 2026-03-31 07:49 UTC
Model: `Qwen/Qwen2.5-1.5B`
Device: `cpu`
Max new tokens: `120`
Max length: `256`
Conflict threshold: `0.55`
Bias scale: `1.50`
Repetition penalty: `1.15`
Frequency penalty: `0.05`
No-repeat ngram size: `3`
Pressure threshold: `0.30`
Max bias gate sum: `1.50`
Entropy threshold: `0.40`
Entropy gamma: `0.10`
Pressure gamma: `0.10`

## Prompt

> You are a vegan chef. Write a detailed weekly meal plan with recipes for each day.

## Summary

- Base lexical score: `1.00`
- Anchor lexical score: `2.00`
- Base positive hits: `1`
- Anchor positive hits: `8`
- Base negative hits: `0`
- Anchor negative hits: `6`
- Anchor protected negative hits: `0`
- Anchor bias active steps: `74`
- Continuations identical: `no`

## First keyword events

- Base first positive: `{'token': 'vegan', 'char_index': 537}`
- Base first negative: `None`
- Anchor first positive: `{'token': 'vegan', 'char_index': 113}`
- Anchor first negative: `{'token': 'egg', 'char_index': 129}`

## Base continuation

The plan should include breakfast, lunch, dinner, and snacks. The meals should be healthy, nutritious, and delicious. The plan should also include any necessary substitutions for ingredients that are not available in your area. The plan should be easy to follow and include a variety of flavors and textures. The plan should also include any necessary cooking techniques or tips for preparing the meals. The plan should be written in a clear and concise style, with each meal listed in order of the day of the week. Monday: 
Breakfast: Vegan Breakfast Burrito
Ingredients:
- 1 cup cooked quinoa

## Anchor-biased continuation

The meals should be healthy, nutritious and include at least one vegetarian dish per week.
Monday: 
Breakfast - Vegan Omelette (Eggs, spinach, mushrooms)
Lunch - Quinoa Salad (Quinoa, cherry tomatoes, cucumber, avocado, feta cheese)
Dinner - Lentil Soup
Tuesday:
Breakfast- Vegan Pancakes (Flour, eggs, maple syrup)
Luncheon- Veggie Stir Fry (Vegetables of your choice stir-fried in soy sauce)
Dinnertime- Grilled Portobello Mushroom Sandwiches on whole wheat bread

