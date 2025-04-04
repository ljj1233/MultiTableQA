import os

import sys
sys.path.append('.')
from Utils.database import DB
from symbolic.utils import choiceGen, corrGen, stmtGen, numericalGen


class University:
    retrieval = [
        ['university_ranking_year', 'ranking_criteria', 'ranking_system', 'university'],
        ['university_ranking_year', 'university'],
        ['university_ranking_year'],
        ['university_ranking_year', 'university'],
        ['university_ranking_year', 'university'],
        ['university_year', 'university'],
        ['university_year', 'university'],
        ['university_year'],
        ['university_year'],
        ['university_year']
    ]
    def __init__(self, dbp) -> None:
        db = DB(dbp)
        self.tables = db.tables

        self.country = self.tables['country']
        self.ranking_system = self.tables['ranking_system']
        self.ranking_criteria = self.tables['ranking_criteria']
        self.university = self.tables['university']
        self.university_ranking_year = self.tables['university_ranking_year']
        self.university_year = self.tables['university_year']

        self.totalScore = self.university_ranking_year.groupby(['university_id', 'year'])['score'].sum().reset_index()
        self.avgScore = self.university_ranking_year.groupby(['university_id', 'year'])['score'].mean().reset_index()


    def q0(self):
        template = 'How many scores do {university_name} get in {year} in {criteria_name}({system_name})?'
        row = self.university_ranking_year.sample(1)
        score = row['score'].iloc[0]
        university_id = row['university_id'].iloc[0]
        year = row['year'].iloc[0]
        ranking_criteria_id = row['ranking_criteria_id'].iloc[0]
        filted = self.ranking_criteria[self.ranking_criteria['id'] == ranking_criteria_id]
        criteria_name = filted['criteria_name'].iloc[0]
        ranking_system_id = filted['ranking_system_id'].iloc[0]
        system_name = self.ranking_system[self.ranking_system['id'] == ranking_system_id]['system_name'].iloc[0]
        university_name = self.university[self.university['id'] == university_id]['university_name'].iloc[0]
        question = template.format(university_name=university_name, year=year, criteria_name=criteria_name, system_name=system_name)

        rightIdx, choices = choiceGen(score, self.university_ranking_year['score'])
        stmts = stmtGen(choices,
                        '{university_name} get <unk> score in {year} in {criteria_name}({system_name}).'.format(
                        university_name=university_name, year=year, criteria_name=criteria_name, system_name=system_name))
        return question, score, rightIdx, choices, stmts

    def q1(self):
        template = 'Which university get the most total score of all ranking criterias in {year}?'
        year = self.totalScore['year'].sample(1).iloc[0]
        filted = self.totalScore[self.totalScore['year'] == year]
        max_score = filted['score'].max()
        filted = filted[filted['score'] == max_score]
        university = self.university[self.university['id'].isin(filted['university_id'])]['university_name'].to_list()
        question = template.format(year=year)

        rightIdx, choices = choiceGen(university, self.university['university_name'])
        stmts = stmtGen(choices,
                        'In {year}, <unk> get the most total score of all ranking criterias.'.format(year=year))
        return question, university, rightIdx, choices, stmts

    def q2(self):
        template = 'How many universities get over {score} total score of all ranking criterias in {year}?'
        year = self.totalScore['year'].sample(1).iloc[0]
        filted = self.totalScore[self.totalScore['year'] == year]
        score = filted['score'].sample(1).iloc[0] - 1
        filted = filted[filted['score'] > score]
        count = len(filted)
        question = template.format(score=score, year=year)

        rightIdx, choices = numericalGen(count)
        stmts = stmtGen(choices,
                        'In {year}, there are <unk> universities get over {score} total score.'.format(score=score, year=year))
        return question, count, rightIdx, choices, stmts

    def q3(self):
        template = 'What is the average total score of all ranking criterias of {university_name} in these years?'
        university_id = self.totalScore['university_id'].sample(1).iloc[0]
        university_name = self.university[self.university['id'] == university_id]['university_name'].iloc[0]
        filted = self.totalScore[self.totalScore['university_id'] == university_id]
        avg = filted['score'].mean()
        question = template.format(university_name=university_name)

        rightIdx, choices = numericalGen(avg)
        stmts = stmtGen(choices,
                        'The average total score of all ranking criterias of {university_name} in these years is <unk>.'.format(university_name=university_name))
        return question, avg, rightIdx, choices, stmts

    def q4(self):
        template = 'What is the total score of all ranking criterias of {university_name} in these years?'
        university_id = self.totalScore['university_id'].sample(1).iloc[0]
        university_name = self.university[self.university['id'] == university_id]['university_name'].iloc[0]
        filted = self.totalScore[self.totalScore['university_id'] == university_id]
        total = filted['score'].sum()
        question = template.format(university_name=university_name)

        rightIdx, choices = numericalGen(total)
        stmts = stmtGen(choices,
                        'The total score of all ranking criterias of {university_name} in these years is <unk>.'.format(university_name=university_name))
        return question, total, rightIdx, choices, stmts

    def q5(self):
        template = 'How many students does {university_name} have in {year}?'
        row = self.university_year.sample(1)
        num_students = row['num_students'].iloc[0]
        university_id = row['university_id'].iloc[0]
        year = row['year'].iloc[0]
        university_name = self.university[self.university['id'] == university_id]['university_name'].iloc[0]
        question = template.format(university_name=university_name, year=year)

        rightIdx, choices = choiceGen(num_students, self.university_year['num_students'])
        stmts = stmtGen(choices,
                        '{university_name} has <unk> students in {year}.'.format(university_name=university_name, year=year))
        return question, num_students, rightIdx, choices, stmts

    def q6(self):
        template = 'Which university get most students in {year}?'
        year = self.university_year['year'].sample(1).iloc[0]
        filted = self.university_year[self.university_year['year'] == year]
        max_student = filted['num_students'].max()
        filted = filted[filted['num_students'] == max_student]
        university = self.university[self.university['id'].isin(filted['university_id'])]['university_name'].to_list()
        question = template.format(year=year)

        rightIdx, choices = choiceGen(university, self.university['university_name'])
        stmts = stmtGen(choices,
                        'In {year}, <unk> get most students.'.format(year=year))
        return question, university, rightIdx, choices, stmts

    def q7(self):
        template = 'How many universities get over {num_students} students in {year}?'
        year = self.university_year['year'].sample(1).iloc[0]
        filted = self.university_year[self.university_year['year'] == year]
        num_students = filted['num_students'].sample(1).iloc[0] - 1
        filted = filted[filted['num_students'] > num_students]
        count = len(filted)
        question = template.format(num_students=num_students, year=year)

        rightIdx, choices = numericalGen(count)
        stmts = stmtGen(choices,
                        'There are <unk> universities get over {num_students} students in {year}.'.format(num_students=num_students, year=year))
        return question, count, rightIdx, choices, stmts

    def q8(self):
        template = 'What is the average student number in all universities in {year}?'
        year = self.university_year['year'].sample(1).iloc[0]
        filted = self.university_year[self.university_year['year'] == year]
        avg = filted['num_students'].mean()
        question = template.format(year=year)

        rightIdx, choices = numericalGen(avg)
        stmts = stmtGen(choices,
                        'The average student number in all universities in {year} is <unk>.'.format(year=year))
        return question, avg, rightIdx, choices, stmts

    def q9(self):
        template = 'What is the total student number in all universities in {year}?'
        year = self.university_year['year'].sample(1).iloc[0]
        filted = self.university_year[self.university_year['year'] == year]
        total = filted['num_students'].sum()
        question = template.format(year=year)

        rightIdx, choices = numericalGen(total)
        stmts = stmtGen(choices,
                        'The total student number in all universities in {year} is <unk>.'.format(year=year))
        return question, total, rightIdx, choices, stmts

    def q10(self):
        template = 'How many students are {university0}({year0}) more than {university1}({year1})?'
        rows = self.university_year.sample(2)
        year0 = rows['year'].iloc[0]
        year1 = rows['year'].iloc[1]
        university0 = self.university[self.university['id'] == rows['university_id'].iloc[0]]['university_name'].iloc[0]
        university1 = self.university[self.university['id'] == rows['university_id'].iloc[1]]['university_name'].iloc[0]
        diff = rows['num_students'].iloc[0] - rows['num_students'].iloc[1]
        question = template.format(university0=university0, year0=year0, university1=university1, year1=year1)

        rightIdx, choices = numericalGen(diff)
        return question, diff, rightIdx, choices

    def q11(self):
        template = 'How many student staff ratio are {university0}({year0}) more than {university1}({year1})?'
        rows = self.university_year.sample(2)
        year0 = rows['year'].iloc[0]
        year1 = rows['year'].iloc[1]
        university0 = self.university[self.university['id'] == rows['university_id'].iloc[0]]['university_name'].iloc[0]
        university1 = self.university[self.university['id'] == rows['university_id'].iloc[1]]['university_name'].iloc[0]
        diff = rows['student_staff_ratio'].iloc[0] - rows['student_staff_ratio'].iloc[1]
        question = template.format(university0=university0, year0=year0, university1=university1, year1=year1)

        rightIdx, choices = numericalGen(diff)
        return question, diff, rightIdx, choices

    def q12(self):
        template = 'What is the correlation between international student percent and female student percent of university-year pairs that have greater or equal than {INT} students?'
        INT = self.university_year['num_students'].sample(1).iloc[0]
        filted = self.university_year[self.university_year['num_students'] >= INT]
        corr = filted['pct_international_students'].corr(filted['pct_female_students'])
        question = template.format(INT=INT)

        rightIdx, choices = corrGen(corr)
        return question, corr, rightIdx, choices

    def q13(self):
        template = 'What is the correlation between international student percent and female student percent of university-year pairs that have less or equal than {INT} students?'
        INT = self.university_year['num_students'].sample(1).iloc[0]
        filted = self.university_year[self.university_year['num_students'] <= INT]
        corr = filted['pct_international_students'].corr(filted['pct_female_students'])
        question = template.format(INT=INT)

        rightIdx, choices = corrGen(corr)
        return question, corr, rightIdx, choices


if __name__ == '__main__':
    dbRoot = 'symDataset/scaledDB/8k/'
    dbn = 'university'
    dbp = os.path.join(dbRoot, dbn, '0.sqlite')
    fi = University(dbp)
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
