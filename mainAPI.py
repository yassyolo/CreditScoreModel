from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('best_gradient_boosting_model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('polyy.pkl')

feature_weights = {
    'NetMonthlyIncome': 2.6,
    'FixedMonthlyExpenses': 2.6,
    'PermanentContractIncome': 1.8,
    'TemporaryContractIncome': 1.5,
    'CivilContractIncome': 1.3,
    'BusinessIncome': 1.7,
    'PensionIncome': 1.0,
    'FreelanceIncome': 1.2,
    'OtherIncome': 1.0,
    'LoanRepayment': 2.0,
    'AccountBalance': 1.5,
    'HasApartmentOrHouse': 2.0,
    'HasLand': 1.5,
    'VehicleCount': 1.0,
    'HasPartialOwnership': 1.5,
    'NoProperty': 1.0,
    'MaritalStatus': 1.5,
    'NumberOfHouseholdMembers': 1.0,
    'MembersWithProvenIncome': 1.5,
    'Dependents': 1.0,
    'IsRetired': 2.0,
    'YearsAtJob': 1.5,
    'MonthsAtJob': 1.0,
    'TotalWorkExperienceYears': 1.5,
    'EducationLevel': 2.0,
    'LoanAmount': 2.0,
    'LoanTermMonths': 1.5,
    'InterestRate': 1.0,
}


def feature_engineering(data):
    data['NetMonthlyIncome'] *= feature_weights['NetMonthlyIncome']
    data['FixedMonthlyExpenses'] *= feature_weights['FixedMonthlyExpenses']
    data['PermanentContractIncome'] *= feature_weights['PermanentContractIncome']
    data['TemporaryContractIncome'] *= feature_weights['TemporaryContractIncome']
    data['CivilContractIncome'] *= feature_weights['CivilContractIncome']
    data['BusinessIncome'] *= feature_weights['BusinessIncome']
    data['PensionIncome'] *= feature_weights['PensionIncome']
    data['FreelanceIncome'] *= feature_weights['FreelanceIncome']
    data['OtherIncome'] *= feature_weights['OtherIncome']
    data['LoanRepayment'] *= feature_weights['LoanRepayment']
    data['AccountBalance'] *= feature_weights['AccountBalance']
    data['HasApartmentOrHouse'] *= feature_weights['HasApartmentOrHouse']
    data['HasLand'] *= feature_weights['HasLand']
    data['VehicleCount'] *= feature_weights['VehicleCount']
    data['HasPartialOwnership'] *= feature_weights['HasPartialOwnership']
    data['NoProperty'] *= feature_weights['NoProperty']
    data['MaritalStatus'] *= feature_weights['MaritalStatus']
    data['NumberOfHouseholdMembers'] *= feature_weights['NumberOfHouseholdMembers']
    data['MembersWithProvenIncome'] *= feature_weights['MembersWithProvenIncome']
    data['Dependents'] *= feature_weights['Dependents']
    data['IsRetired'] *= feature_weights['IsRetired']
    data['YearsAtJob'] *= feature_weights['YearsAtJob']
    data['MonthsAtJob'] *= feature_weights['MonthsAtJob']
    data['TotalWorkExperienceYears'] *= feature_weights['TotalWorkExperienceYears']
    data['EducationLevel'] *= feature_weights['EducationLevel']
    data['LoanAmount'] *= feature_weights['LoanAmount']
    data['LoanTermMonths'] *= feature_weights['LoanTermMonths']
    data['InterestRate'] *= feature_weights['InterestRate']
    data['DisposableIncome'] = data['NetMonthlyIncome'] - (data['FixedMonthlyExpenses'] + data['LoanRepayment'])
    data['LoanRepaymentBurden'] = data['LoanRepayment'] / (data['NetMonthlyIncome'] + 1)
    data['AccountBalanceToLoanRatio'] = data['AccountBalance'] / (data['LoanAmount'] + 1)
    data['AccountBalanceToIncomeRatio'] = data['AccountBalance'] / (data['NetMonthlyIncome'] + 1)

    data['WeightedIncome'] = (data['PermanentContractIncome'] * 1.5 +
                              data['TemporaryContractIncome'] * 1.2 +
                              data['CivilContractIncome'] * 1.1 +
                              data['BusinessIncome'] * 1.3 +
                              data['PensionIncome'] * 1.0 +
                              data['FreelanceIncome'] * 1.1 +
                              data['OtherIncome'] * 1.0)

    data['PropertyScore'] = (data['HasApartmentOrHouse'] * 3 +
                             data['HasLand'] * 2 +
                             data['VehicleCount'] * 1 +
                             data['HasPartialOwnership'] * 1.5)

    data['HouseholdScore'] = (data['NumberOfHouseholdMembers'] - data['Dependents'] +
                              data['MembersWithProvenIncome'] * 2)

    data['IncomeLoanAmountInteraction'] = data['NetMonthlyIncome'] * data['LoanAmount']
    data['ExpensesDisposableIncomeInteraction'] = data['FixedMonthlyExpenses'] * data['DisposableIncome']
    data['DependencyRatio'] = data['Dependents'] / (data['NumberOfHouseholdMembers'] + 1)
    data['ProvenIncomeProportion'] = data['MembersWithProvenIncome'] / (data['NumberOfHouseholdMembers'] + 1)
    data['JobExperienceRatio'] = data['MonthsAtJob'] / (data['TotalWorkExperienceMonths'] + 1)
    data['EducationIncomeInteraction'] = data['EducationLevel'] * data['NetMonthlyIncome']
    data['IncomeToDebtRatio'] = data['NetMonthlyIncome'] / (data['LoanRepayment'] + data['FixedMonthlyExpenses'] + 1)
    data['SavingsRate'] = data['AccountBalance'] / (data['NetMonthlyIncome'] + 1)
    data['PropertyVehicleScore'] = data['HasApartmentOrHouse'] * 3 + data['HasLand'] * 2 + data['VehicleCount'] * 1.5
    data['EffectiveLoanTerm'] = data['LoanTermMonths'] / (data['NetMonthlyIncome'] + 1)
    data['LoanTermPropertyInteraction'] = data['LoanTermMonths'] * data['PropertyScore']
    data['JobEducationInteraction'] = data['JobExperienceRatio'] * data['EducationLevel']
    data['IncomeVariability'] = data['NetMonthlyIncome'] - data[
        ['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome', 'PensionIncome',
         'FreelanceIncome', 'OtherIncome']].max(axis=1)
    data['TotalIncome'] = data[
        ['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome', 'PensionIncome',
         'FreelanceIncome', 'OtherIncome']].sum(axis=1)
    data['LoanRepaymentIncomeDebtInteraction'] = data['LoanRepayment'] * data['IncomeToDebtRatio']
    data['NetIncomePerHouseholdMember'] = data['NetMonthlyIncome'] / (data['NumberOfHouseholdMembers'] + 1)
    data['DebtPerHouseholdMember'] = (
                data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) / (data['NumberOfHouseholdMembers'] + 1))
    data['SavingsPerHouseholdMember'] = data['AccountBalance'] / (data['NumberOfHouseholdMembers'] + 1)
    data['DebtToIncomeRatio'] = (
                data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) / data['NetMonthlyIncome'] + 1)
    data['DebtServiceCoverageRatio'] = data['NetMonthlyIncome'] / (
                data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) + 1)
    data['IncomeConsistencyScore'] = data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome',
                                           'BusinessIncome', 'PensionIncome', 'FreelanceIncome', 'OtherIncome']].std(
        axis=1) / data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome', 'BusinessIncome',
                        'PensionIncome', 'FreelanceIncome', 'OtherIncome']].mean(axis=1)
    data['StableIncomeScore'] = data[['PermanentContractIncome', 'TemporaryContractIncome', 'CivilContractIncome',
                                      'BusinessIncome', 'PensionIncome', 'FreelanceIncome', 'OtherIncome']].max(
        axis=1) / data['NetMonthlyIncome']
    data['IncomeAndDebtInteraction'] = (data['NetMonthlyIncome'] ** 2) / (
                data[['LoanRepayment', 'FixedMonthlyExpenses']].sum(axis=1) + 1)
    data['IncomeSavingsInteraction'] = data['NetMonthlyIncome'] * data['SavingsRate']
    data['TotalLoanCost'] = data['LoanAmount'] * (1 + (data['InterestRate'] / 100) * (data['LoanTermMonths'] / 12))
    data['InterestRateToIncomeRatio'] = data['InterestRate'] / (data['NetMonthlyIncome'] + 1)
    data['HasOtherCreditsImpact'] = data['HasOtherCredits'] * data['IncomeToDebtRatio']

    poly_features = poly.transform(data[['NetMonthlyIncome', 'LoanRepaymentBurden',
                                         'PropertyScore', 'HouseholdScore',
                                         'AccountBalanceToLoanRatio', 'AccountBalanceToIncomeRatio',
                                         'IncomeLoanAmountInteraction', 'ExpensesDisposableIncomeInteraction',
                                         'DependencyRatio', 'ProvenIncomeProportion', 'JobExperienceRatio',
                                         'EducationIncomeInteraction', 'IncomeToDebtRatio', 'SavingsRate',
                                         'PropertyVehicleScore', 'EffectiveLoanTerm', 'LoanTermPropertyInteraction',
                                         'JobEducationInteraction', 'IncomeVariability', 'TotalIncome',
                                         'LoanRepaymentIncomeDebtInteraction', 'NetIncomePerHouseholdMember',
                                         'DebtPerHouseholdMember', 'SavingsPerHouseholdMember',
                                         'DebtToIncomeRatio', 'DebtServiceCoverageRatio',
                                         'IncomeConsistencyScore', 'StableIncomeScore',
                                         'IncomeAndDebtInteraction', 'IncomeSavingsInteraction',
                                         'TotalLoanCost', 'InterestRateToIncomeRatio', 'HasOtherCreditsImpact']])

    poly_feature_names = poly.get_feature_names_out(['NetMonthlyIncome', 'LoanRepaymentBurden',
                                                     'PropertyScore', 'HouseholdScore',
                                                     'AccountBalanceToLoanRatio', 'AccountBalanceToIncomeRatio',
                                                     'IncomeLoanAmountInteraction',
                                                     'ExpensesDisposableIncomeInteraction',
                                                     'DependencyRatio', 'ProvenIncomeProportion', 'JobExperienceRatio',
                                                     'EducationIncomeInteraction', 'IncomeToDebtRatio', 'SavingsRate',
                                                     'PropertyVehicleScore', 'EffectiveLoanTerm',
                                                     'LoanTermPropertyInteraction',
                                                     'JobEducationInteraction', 'IncomeVariability', 'TotalIncome',
                                                     'LoanRepaymentIncomeDebtInteraction',
                                                     'NetIncomePerHouseholdMember',
                                                     'DebtPerHouseholdMember', 'SavingsPerHouseholdMember',
                                                     'DebtToIncomeRatio', 'DebtServiceCoverageRatio',
                                                     'IncomeConsistencyScore', 'StableIncomeScore',
                                                     'IncomeAndDebtInteraction', 'IncomeSavingsInteraction',
                                                     'TotalLoanCost', 'InterestRateToIncomeRatio',
                                                     'HasOtherCreditsImpact'])

    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    data = pd.concat([data, poly_df], axis=1)
    return data


def calculate_credit_score(probability, min_score=300, max_score=850):
    probability = np.clip(probability, 0, 1)
    credit_score = min_score + (max_score - min_score) * probability
    return credit_score


def assess_risk(probability, credit_score, min_score=300, max_score=850):
    risk_categories = {
        'Low': (min_score + 0.7 * (max_score - min_score), max_score),
        'Medium': (min_score + 0.4 * (max_score - min_score), min_score + 0.7 * (max_score - min_score)),
        'High': (min_score, min_score + 0.4 * (max_score - min_score))
    }
    for category, (low_threshold, high_threshold) in risk_categories.items():
        if low_threshold <= credit_score <= high_threshold:
            return category
    return 'Unknown'



with open('feature_columns.pkl', 'rb') as f:
    feature_columns = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    df = feature_engineering(df)
    df_scaled = scaler.transform(df)

    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        return jsonify({
            'error': 'Missing columns',
            'expected_columns': feature_columns,
            'missing_columns': missing_columns
        })

    prediction = model.predict_proba(df_scaled)
    probability_positive_class = prediction[0, 1]

    credit_score = calculate_credit_score(probability_positive_class)
    risk_category = assess_risk(probability_positive_class, credit_score)

    return jsonify({
        'probability': probability_positive_class,
        'credit_score': credit_score,
        'risk_category': risk_category
    })
if __name__ == '__main__':
    app.run(debug=True)