import os
import random
from unicodedata import numeric
import pandas as pd

import sys
sys.path.append('.')
from Utils.database import DB
from symbolic.utils import choiceGen, corrGen, stmtGen, numericalGen


class GlobalBiodiversity:
    retrieval = [
        ['global_biod_species_extinction_risks'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range'],
        ['global_biod_species_extinction_risks'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range'],
        ['global_biod_species_extinction_risks', 'global_biod_species_occ_endemism_and_small_range']
    ]
    def __init__(self, dbp) -> None:
        db = DB(dbp)
        self.tables = db.tables

        self.risks = self.tables['global_biod_species_extinction_risks']
        self.occ = self.tables['global_biod_species_occ_endemism_and_small_range']
        self.risks['popden'] = self.risks['popden'].astype(float)

        self.merged_df = pd.merge(self.risks, self.occ, left_on='species', right_on='species')


    def q0(self):
        template = 'What is the lcat of {species}?'
        row = self.risks.sample(1)
        species = row['species'].iloc[0]
        lcat = row['lcat'].iloc[0]
        question = template.format(species=species)

        rightIdx, choices = choiceGen(lcat, self.risks['lcat'])
        stmts = stmtGen(choices,
                        'The lcat of {species} is <unk>.'.format(species=species))
        return question, lcat, rightIdx, choices, stmts

    def q5(self):
        template = 'What is the population density of {species}?'
        row = self.risks.sample(1)
        species = row['species'].iloc[0]
        popden = row['popden'].iloc[0]
        question = template.format(species=species)

        rightIdx, choices = choiceGen(popden, self.risks['popden'])
        stmts = stmtGen(choices,
                        'The population density of {species} is <unk>.'.format(species=species))
        return question, popden, rightIdx, choices, stmts

    def q1(self):
        template = 'Which species in {kingdom} kingdom has most population density?'
        kingdom = self.merged_df.sample(1)['kingdom'].iloc[0]
        filted = self.merged_df[self.merged_df['kingdom'] == kingdom]
        max_val = filted['popden'].max()
        species = filted[filted['popden'] == max_val]['species'].to_list()
        question = template.format(kingdom=kingdom)

        rightIdx, choices = choiceGen(species, self.merged_df['species'])
        stmts = stmtGen(choices,
                        'In {kingdom} kingdom, <unk> has most population density.'.format(kingdom=kingdom))
        return question, species, rightIdx, choices, stmts

    def q6(self):
        template = 'Which species in {phylum} phylum has most population density?'
        phylum = self.merged_df.sample(1)['phylum'].iloc[0]
        filted = self.merged_df[self.merged_df['phylum'] == phylum]
        max_val = filted['popden'].max()
        species = filted[filted['popden'] == max_val]['species'].to_list()
        question = template.format(phylum=phylum)

        rightIdx, choices = choiceGen(species, self.merged_df['species'])
        stmts = stmtGen(choices,
                        'In {phylum} phylum, <unk> has most population density.'.format(phylum=phylum))
        return question, species, rightIdx, choices, stmts

    def q2(self):
        template = 'How many {kingdom} kingdom species have greater or equal than {REAL} population density?'
        kingdom = self.merged_df.sample(1)['kingdom'].iloc[0]
        filted = self.merged_df[self.merged_df['kingdom'] == kingdom]
        REAL = filted.sample(1)['popden'].iloc[0]
        filted = filted[filted['popden'] >= REAL]
        count = len(filted)
        question = template.format(kingdom=kingdom, REAL=REAL)

        rightIdx, choices = numericalGen(count)
        stmts = stmtGen(choices,
                        'There are <unk> {kingdom} kingdom species have greater or equal than {REAL} population density.'.format(kingdom=kingdom, REAL=REAL))
        return question, count, rightIdx, choices, stmts

    def q7(self):
        template = 'How many {phylum} phylum species have more than {REAL} population density?'
        phylum = self.merged_df.sample(1)['phylum'].iloc[0]
        filted = self.merged_df[self.merged_df['phylum'] == phylum]
        REAL = filted.sample(1)['popden'].iloc[0]
        filted = filted[filted['popden'] >= REAL]
        count = len(filted)
        question = template.format(phylum=phylum, REAL=REAL)

        rightIdx, choices = numericalGen(count)
        stmts = stmtGen(choices,
                        'There are <unk> {phylum} kingdom species have greater or equal than {REAL} population density.'.format(phylum=phylum, REAL=REAL))
        return question, count, rightIdx, choices, stmts

    def q3(self):
        template = 'What is the average population density of {cls} class species?'
        cls = self.merged_df.sample(1)['class'].iloc[0]
        filted = self.merged_df[self.merged_df['class'] == cls]
        avg = filted['popden'].mean()
        question = template.format(cls=cls)

        rightIdx, choices = numericalGen(avg)
        stmts = stmtGen(choices,
                        'The average population density of {cls} class species is <unk>.'.format(cls=cls))
        return question, avg, rightIdx, choices, stmts

    def q4(self):
        template = 'What is the total population density of {cls} class species?'
        cls = self.merged_df.sample(1)['class'].iloc[0]
        filted = self.merged_df[self.merged_df['class'] == cls]
        total = filted['popden'].sum()
        question = template.format(cls=cls)

        rightIdx, choices = numericalGen(total)
        stmts = stmtGen(choices,
                        'The total population density of {cls} class species is <unk>.'.format(cls=cls))
        return question, total, rightIdx, choices, stmts

    def q8(self):
        template = 'What is the average population density of {order} order species?'
        order = self.merged_df.sample(1)['order'].iloc[0]
        filted = self.merged_df[self.merged_df['order'] == order]
        avg = filted['popden'].mean()
        question = template.format(order=order)

        rightIdx, choices = numericalGen(avg)
        stmts = stmtGen(choices,
                        'The average population density of {order} order species is <unk>.'.format(order=order))
        return question, avg, rightIdx, choices, stmts

    def q9(self):
        template = 'What is the total population density of {order} order species?'
        order = self.merged_df.sample(1)['order'].iloc[0]
        filted = self.merged_df[self.merged_df['order'] == order]
        total = filted['popden'].sum()
        question = template.format(order=order)

        rightIdx, choices = numericalGen(total)
        stmts = stmtGen(choices,
                        'The total population density of {order} order species is <unk>.'.format(order=order))
        return question, total, rightIdx, choices, stmts

    def q10(self):
        template = 'How many pthreats are {species0} more than {species1}?'
        rows = self.risks.sample(2)
        species0 = rows['species'].iloc[0]
        species1 = rows['species'].iloc[1]
        diff = rows['pthreat'].iloc[0] - rows['pthreat'].iloc[1]
        question = template.format(species0=species0, species1=species1)

        rightIdx, choices = numericalGen(diff)
        return question, diff, rightIdx, choices

    def q11(self):
        template = 'How many population densities are {species0} more than {species1}?'
        rows = self.risks[self.risks['popden'].notna()].sample(2)
        species0 = rows['species'].iloc[0]
        species1 = rows['species'].iloc[1]
        diff = rows['popden'].iloc[0] - rows['popden'].iloc[1]
        question = template.format(species0=species0, species1=species1)

        rightIdx, choices = numericalGen(diff)
        return question, diff, rightIdx, choices

    def q12(self):
        template = 'What is the correlation between popden and parkpct of {kingdom} species?'
        kingdom = self.occ['kingdom'].dropna().sample(1).iloc[0]
        filted = self.risks[self.risks['species'].isin(self.occ[self.occ['kingdom'] == kingdom]['species'])]
        corr = filted['popden'].corr(filted['parkpct'])
        question = template.format(kingdom=kingdom)

        rightIdx, choices = corrGen(corr)
        return question, corr, rightIdx, choices

    def q13(self):
        template = 'What is the correlation between popden and pthreat of {kingdom} species?'
        kingdom = self.occ['kingdom'].dropna().sample(1).iloc[0]
        filted = self.risks[self.risks['species'].isin(self.occ[self.occ['kingdom'] == kingdom]['species'])]
        corr = filted['popden'].corr(filted['pthreat'])
        question = template.format(kingdom=kingdom)

        rightIdx, choices = corrGen(corr)
        return question, corr, rightIdx, choices


if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB/8k/'
    dbn = 'global_biodiversity'
    dbp = os.path.join(dbRoot, dbn, '0.sqlite')
    fi = GlobalBiodiversity(dbp)
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
