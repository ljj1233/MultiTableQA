import os
import random
import pandas as pd

import sys

sys.path.append(".")
from Utils.database import DB
from symbolic.utils import choiceGen, stmtGen, numericalGen, corrGen


class Cookbook:
    retrieval = [
        ["Nutrition", "Recipe"],
        ["Nutrition", "Recipe"],
        ["Nutrition", "Recipe"],
        ["Nutrition"],
        ["Nutrition"],
        ["Nutrition", "Recipe"],
        ["Nutrition", "Recipe"],
        ["Nutrition", "Recipe"],
        ["Nutrition"],
        ["Nutrition"],
    ]

    def __init__(self, dbp) -> None:
        db = DB(dbp)
        self.tables = db.tables

        self.ingredient = self.tables["Ingredient"]
        self.recipe = self.tables["Recipe"]
        self.nutrition = self.tables["Nutrition"]
        self.quantity = self.tables["Quantity"]

        self.merged_df = pd.merge(
            self.nutrition, self.recipe, left_on="recipe_id", right_on="recipe_id"
        )

    def q0(self):
        template = "How many calories are there in the {title}?"
        row = self.nutrition.sample(1)
        recipe_id = row["recipe_id"].iloc[0]
        calories = row["calories"].iloc[0]
        title = self.recipe[self.recipe["recipe_id"] == recipe_id]["title"].iloc[0]
        question = template.format(title=title)

        rightIdx, choices = choiceGen(calories, self.nutrition["calories"])
        stmts = stmtGen(
            choices, "There are <unk> calories in the {title}.".format(title=title)
        )
        return question, calories, rightIdx, choices, stmts

    def q1(self):
        template = "Which recipe has most {COL} per serving?"
        COL = random.choice("protein/carbo/total_fat/sat_fat".split("/"))
        self.merged_df["COL_per_serving"] = (
            self.merged_df[COL] / self.merged_df["servings"]
        )
        max_val = self.merged_df["COL_per_serving"].max()
        items = self.merged_df[self.merged_df["COL_per_serving"] == max_val][
            "title"
        ].to_list()
        question = template.format(COL=COL)

        rightIdx, choices = choiceGen(items, self.merged_df["title"])
        stmts = stmtGen(choices, "<unk> has most {COL} per serving.".format(COL=COL))
        return question, items, rightIdx, choices, stmts

    def q2(self):
        template = "How many recipes have more than {REAL:.2f} calories?"
        calories = self.merged_df.sample(1)["calories"].iloc[0] - 0.01
        filted = self.merged_df[self.merged_df["calories"] > calories]
        count = len(filted)
        question = template.format(REAL=calories)

        rightIdx, choices = numericalGen(count)
        stmts = stmtGen(
            choices,
            "There are <unk> recipes have more than {REAL:.2f} calories.".format(
                REAL=calories
            ),
        )
        return question, count, rightIdx, choices, stmts

    def q3(self):
        template = "What is the average calories of recipies that have greater or equal than {REAL} proteins?"
        protein = self.nutrition.sample(1)["protein"].iloc[0]
        filted = self.nutrition[self.nutrition["protein"] >= protein]
        avg = filted["calories"].mean()
        question = template.format(REAL=protein)

        rightIdx, choices = numericalGen(avg)
        stmts = stmtGen(
            choices,
            "The average calories of recipies that have greater or equal than {REAL} proteins is <unk>.".format(
                REAL=protein
            ),
        )
        return question, avg, rightIdx, choices, stmts

    def q4(self):
        template = "What is the total calories of recipies that have greater or equal than {REAL} proteins?"
        protein = self.nutrition.sample(1)["protein"].iloc[0]
        filted = self.nutrition[self.nutrition["protein"] >= protein]
        total = filted["calories"].sum()
        question = template.format(REAL=protein)

        rightIdx, choices = numericalGen(total)
        stmts = stmtGen(
            choices,
            "The total calories of recipies that have greater or equal than {REAL} proteins is <unk>.".format(
                REAL=protein
            ),
        )
        return question, total, rightIdx, choices, stmts

    def q5(self):
        template = "How many proteins are there in the {title}?"
        row = self.nutrition.sample(1)
        recipe_id = row["recipe_id"].iloc[0]
        protein = row["protein"].iloc[0]
        title = self.recipe[self.recipe["recipe_id"] == recipe_id]["title"].iloc[0]
        question = template.format(title=title)

        rightIdx, choices = choiceGen(protein, self.nutrition["protein"])
        stmts = stmtGen(
            choices, "There are <unk> proteins in the {title}.".format(title=title)
        )
        return question, protein, rightIdx, choices, stmts

    def q6(self):
        template = "Which recipe has most {COL}?"
        COL = random.choice("protein/carbo/total_fat/sat_fat".split("/"))
        max_val = self.merged_df[COL].max()
        items = self.merged_df[self.merged_df[COL] == max_val]["title"].to_list()
        question = template.format(COL=COL)

        rightIdx, choices = choiceGen(items, self.merged_df["title"])
        stmts = stmtGen(choices, "<unk> has most {COL}.".format(COL=COL))
        return question, items, rightIdx, choices, stmts

    def q7(self):
        template = "How many recipes have more than {REAL:.2f} proteins?"
        protein = self.merged_df.sample(1)["protein"].iloc[0] - 0.01
        filted = self.merged_df[self.merged_df["protein"] > protein]
        count = len(filted)
        question = template.format(REAL=protein)

        rightIdx, choices = numericalGen(count)
        stmts = stmtGen(
            choices,
            "There are <unk> recipes have more than {REAL:.2f} proteins.".format(
                REAL=protein
            ),
        )
        return question, count, rightIdx, choices, stmts

    def q8(self):
        template = "What is the average proteins of recipies that have greater or equal than {REAL} calories?"
        calories = self.nutrition.sample(1)["calories"].iloc[0]
        filted = self.nutrition[self.nutrition["calories"] >= calories]
        avg = filted["protein"].mean()
        question = template.format(REAL=calories)

        rightIdx, choices = numericalGen(avg)
        stmts = stmtGen(
            choices,
            "The average proteins of recipies that have greater or equal than {REAL} calories is <unk>.".format(
                REAL=calories
            ),
        )
        return question, avg, rightIdx, choices, stmts

    def q9(self):
        template = "What is the total proteins of recipies that have greater or equal than {REAL} calories?"
        calories = self.nutrition.sample(1)["calories"].iloc[0]
        filted = self.nutrition[self.nutrition["calories"] >= calories]
        total = filted["protein"].sum()
        question = template.format(REAL=calories)

        rightIdx, choices = numericalGen(total)
        stmts = stmtGen(
            choices,
            "The total proteins of recipies that have greater or equal than {REAL} calories is <unk>.".format(
                REAL=calories
            ),
        )
        return question, total, rightIdx, choices, stmts

    def q10(self):
        template = "How many preparation minutes are {title0} more than {title1}?"
        rows = self.recipe.sample(2)
        title0 = rows["title"].iloc[0]
        title1 = rows["title"].iloc[1]
        diff = rows["prep_min"].iloc[0] - rows["prep_min"].iloc[1]
        question = template.format(title0=title0, title1=title1)

        rightIdx, choices = numericalGen(diff)
        return question, diff, rightIdx, choices

    def q11(self):
        template = "How many cook minutes are {title0} more than {title1}?"
        rows = self.recipe.sample(2)
        title0 = rows["title"].iloc[0]
        title1 = rows["title"].iloc[1]
        diff = rows["cook_min"].iloc[0] - rows["cook_min"].iloc[1]
        question = template.format(title0=title0, title1=title1)

        rightIdx, choices = numericalGen(diff)
        return question, diff, rightIdx, choices

    def q12(self):
        template = "What is the correlation between total fat and sat fat of recipies that have greater or equal than {REAL:.2f} calories?"
        calories = self.nutrition.sample(1)["calories"].iloc[0]
        filted = self.nutrition[self.nutrition["calories"] >= calories]
        corr = filted["total_fat"].corr(filted["sat_fat"])
        question = template.format(REAL=calories)

        rightIdx, choices = corrGen(corr)
        return question, corr, rightIdx, choices

    def q13(self):
        template = "What is the correlation between vitamin C and vitamin A of recipies that have greater or equal than {REAL:.2f} calories?"
        calories = self.nutrition.sample(1)["calories"].iloc[0]
        filted = self.nutrition[self.nutrition["calories"] >= calories]
        corr = filted["vitamin_c"].corr(filted["vitamin_a"])
        question = template.format(REAL=calories)

        rightIdx, choices = corrGen(corr)
        return question, corr, rightIdx, choices


if __name__ == "__main__":
    dbRoot = "symDataset/scaledDB/8k/"
    dbn = "cookbook"
    dbp = os.path.join(dbRoot, dbn, f"0.sqlite")
    fi = Cookbook(dbp)
    print(fi.q0())
    print(fi.q1())
    print(fi.q2())
    print(fi.q3())
    print(fi.q4())
    print(fi.q5())
    print(fi.q6())
    print(fi.q7())
    print(fi.q8())
    print(fi.q9())
    print(fi.q10())
    print(fi.q11())
    print(fi.q12())
    print(fi.q13())
