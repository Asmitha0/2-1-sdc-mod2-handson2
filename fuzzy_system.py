# fuzzy_system.py
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

class FuzzyWineQualitySystem:
    def __init__(self):
        """Initialize the fuzzy wine quality prediction system"""
        self._create_variables()
        self._define_membership_functions()
        self._define_rules()
        self._create_control_system()
        
    def _create_variables(self):
        """Create input and output variables"""
        # Input variables
        self.alcohol = ctrl.Antecedent(np.arange(8, 15, 0.1), 'alcohol')
        self.volatile_acidity = ctrl.Antecedent(np.arange(0, 1.5, 0.01), 'volatile_acidity')
        self.citric_acid = ctrl.Antecedent(np.arange(0, 0.8, 0.01), 'citric_acid')
        self.residual_sugar = ctrl.Antecedent(np.arange(0, 12, 0.1), 'residual_sugar')
        self.chlorides = ctrl.Antecedent(np.arange(0, 0.3, 0.001), 'chlorides')
        self.free_sulfur_dioxide = ctrl.Antecedent(np.arange(0, 70, 0.5), 'free_sulfur_dioxide')
        self.total_sulfur_dioxide = ctrl.Antecedent(np.arange(0, 250, 1), 'total_sulfur_dioxide')
        self.density = ctrl.Antecedent(np.arange(0.985, 1.005, 0.0001), 'density')
        self.ph = ctrl.Antecedent(np.arange(2.5, 4.5, 0.01), 'ph')
        self.sulphates = ctrl.Antecedent(np.arange(0, 1.2, 0.01), 'sulphates')
        
        # Output variable
        self.quality = ctrl.Consequent(np.arange(0, 10, 0.1), 'quality')
        
    def _define_membership_functions(self):
        """Define membership functions for all variables"""
        
        # Alcohol (g/L) - Low, Medium, High
        self.alcohol['low'] = fuzz.trimf(self.alcohol.universe, [8, 8, 10.5])
        self.alcohol['medium'] = fuzz.trimf(self.alcohol.universe, [9.5, 11.5, 13])
        self.alcohol['high'] = fuzz.trimf(self.alcohol.universe, [12, 14, 14])
        
        # Volatile Acidity (g/L) - Low, Medium, High
        self.volatile_acidity['low'] = fuzz.trimf(self.volatile_acidity.universe, [0, 0, 0.4])
        self.volatile_acidity['medium'] = fuzz.trimf(self.volatile_acidity.universe, [0.3, 0.6, 0.8])
        self.volatile_acidity['high'] = fuzz.trimf(self.volatile_acidity.universe, [0.7, 1.1, 1.5])
        
        # Citric Acid (g/L) - Low, Medium, High
        self.citric_acid['low'] = fuzz.trimf(self.citric_acid.universe, [0, 0, 0.2])
        self.citric_acid['medium'] = fuzz.trimf(self.citric_acid.universe, [0.15, 0.35, 0.5])
        self.citric_acid['high'] = fuzz.trimf(self.citric_acid.universe, [0.4, 0.6, 0.8])
        
        # Residual Sugar (g/L) - Low, Medium, High
        self.residual_sugar['low'] = fuzz.trimf(self.residual_sugar.universe, [0, 0, 3])
        self.residual_sugar['medium'] = fuzz.trimf(self.residual_sugar.universe, [2, 5, 8])
        self.residual_sugar['high'] = fuzz.trimf(self.residual_sugar.universe, [6, 10, 12])
        
        # Chlorides (g/L) - Low, Medium, High
        self.chlorides['low'] = fuzz.trimf(self.chlorides.universe, [0, 0, 0.08])
        self.chlorides['medium'] = fuzz.trimf(self.chlorides.universe, [0.06, 0.13, 0.2])
        self.chlorides['high'] = fuzz.trimf(self.chlorides.universe, [0.16, 0.25, 0.3])
        
        # Free Sulfur Dioxide (mg/L) - Low, Medium, High
        self.free_sulfur_dioxide['low'] = fuzz.trimf(self.free_sulfur_dioxide.universe, [0, 0, 15])
        self.free_sulfur_dioxide['medium'] = fuzz.trimf(self.free_sulfur_dioxide.universe, [10, 30, 50])
        self.free_sulfur_dioxide['high'] = fuzz.trimf(self.free_sulfur_dioxide.universe, [40, 60, 70])
        
        # Total Sulfur Dioxide (mg/L) - Low, Medium, High
        self.total_sulfur_dioxide['low'] = fuzz.trimf(self.total_sulfur_dioxide.universe, [0, 0, 60])
        self.total_sulfur_dioxide['medium'] = fuzz.trimf(self.total_sulfur_dioxide.universe, [50, 120, 180])
        self.total_sulfur_dioxide['high'] = fuzz.trimf(self.total_sulfur_dioxide.universe, [150, 220, 250])
        
        # Density (g/cm³) - Low, Medium, High
        self.density['low'] = fuzz.trimf(self.density.universe, [0.985, 0.99, 0.995])
        self.density['medium'] = fuzz.trimf(self.density.universe, [0.993, 0.997, 1.0])
        self.density['high'] = fuzz.trimf(self.density.universe, [0.998, 1.0, 1.005])
        
        # pH - Low, Medium, High
        self.ph['low'] = fuzz.trimf(self.ph.universe, [2.5, 2.8, 3.2])
        self.ph['medium'] = fuzz.trimf(self.ph.universe, [3.0, 3.4, 3.8])
        self.ph['high'] = fuzz.trimf(self.ph.universe, [3.6, 4.0, 4.5])
        
        # Sulphates (g/L) - Low, Medium, High
        self.sulphates['low'] = fuzz.trimf(self.sulphates.universe, [0, 0, 0.4])
        self.sulphates['medium'] = fuzz.trimf(self.sulphates.universe, [0.3, 0.6, 0.8])
        self.sulphates['high'] = fuzz.trimf(self.sulphates.universe, [0.7, 1.0, 1.2])
        
        # Quality output - Very Poor, Poor, Average, Good, Excellent
        self.quality['very_poor'] = fuzz.trimf(self.quality.universe, [0, 0, 2.5])
        self.quality['poor'] = fuzz.trimf(self.quality.universe, [2, 4, 5])
        self.quality['average'] = fuzz.trimf(self.quality.universe, [4, 6, 7])
        self.quality['good'] = fuzz.trimf(self.quality.universe, [6, 7.5, 9])
        self.quality['excellent'] = fuzz.trimf(self.quality.universe, [8, 9.5, 10])
        
    def _define_rules(self):
        """Define fuzzy rules based on wine chemistry knowledge"""
        
        # Rule 1: Higher alcohol and lower volatile acidity -> Good quality
        rule1 = ctrl.Rule(
            self.alcohol['high'] & self.volatile_acidity['low'], 
            self.quality['good']
        )
        
        # Rule 2: High citric acid and moderate pH -> Good quality
        rule2 = ctrl.Rule(
            self.citric_acid['high'] & self.ph['medium'], 
            self.quality['good']
        )
        
        # Rule 3: Low residual sugar and low chlorides -> Good quality
        rule3 = ctrl.Rule(
            self.residual_sugar['low'] & self.chlorides['low'], 
            self.quality['good']
        )
        
        # Rule 4: High alcohol and high sulphates -> Excellent quality
        rule4 = ctrl.Rule(
            self.alcohol['high'] & self.sulphates['high'], 
            self.quality['excellent']
        )
        
        # Rule 5: Low alcohol and high volatile acidity -> Poor quality
        rule5 = ctrl.Rule(
            self.alcohol['low'] & self.volatile_acidity['high'], 
            self.quality['poor']
        )
        
        # Rule 6: High residual sugar and low acidity -> Poor quality
        rule6 = ctrl.Rule(
            self.residual_sugar['high'] & self.citric_acid['low'], 
            self.quality['poor']
        )
        
        # Rule 7: Moderate alcohol and balanced acidity -> Average quality
        rule7 = ctrl.Rule(
            self.alcohol['medium'] & self.citric_acid['medium'] & self.volatile_acidity['medium'], 
            self.quality['average']
        )
        
        # Rule 8: High alcohol and moderate all characteristics -> Good quality
        rule8 = ctrl.Rule(
            self.alcohol['high'] & self.citric_acid['medium'] & self.sulphates['medium'], 
            self.quality['good']
        )
        
        # Rule 9: Very high volatile acidity -> Very poor quality
        rule9 = ctrl.Rule(
            self.volatile_acidity['high'] & self.alcohol['low'], 
            self.quality['very_poor']
        )
        
        # Rule 10: High total SO2 and high free SO2 -> Average quality (preservation)
        rule10 = ctrl.Rule(
            self.total_sulfur_dioxide['high'] & self.free_sulfur_dioxide['high'], 
            self.quality['average']
        )
        
        # Rule 11: Excellent balance of all factors -> Excellent quality
        rule11 = ctrl.Rule(
            self.alcohol['high'] & self.citric_acid['high'] & 
            self.sulphates['high'] & self.volatile_acidity['low'],
            self.quality['excellent']
        )
        
        # Rule 12: Poor all around -> Very poor quality
        rule12 = ctrl.Rule(
            self.alcohol['low'] & self.volatile_acidity['high'] & 
            self.chlorides['high'] & self.citric_acid['low'],
            self.quality['very_poor']
        )
        
        # Rule 13: Good alcohol and low density -> Good quality
        rule13 = ctrl.Rule(
            self.alcohol['high'] & self.density['low'],
            self.quality['good']
        )
        
        # Rule 14: High pH and low acidity -> Poor quality (instability)
        rule14 = ctrl.Rule(
            self.ph['high'] & self.citric_acid['low'],
            self.quality['poor']
        )
        
        # Rule 15: Balanced pH and moderate everything -> Average quality
        rule15 = ctrl.Rule(
            self.ph['medium'] & self.alcohol['medium'] & 
            self.sulphates['medium'] & self.citric_acid['medium'],
            self.quality['average']
        )
        
        # Collect all rules
        self.rules = [
            rule1, rule2, rule3, rule4, rule5, rule6, rule7, 
            rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15
        ]
        
    def _create_control_system(self):
        """Create the control system and simulation"""
        self.quality_ctrl = ctrl.ControlSystem(self.rules)
        self.quality_sim = ctrl.ControlSystemSimulation(self.quality_ctrl)
        
    def predict(self, features):
        """
        Predict wine quality based on input features
        
        Parameters:
        features (dict): Dictionary containing wine chemical properties
        
        Returns:
        float: Predicted quality score (0-10)
        """
        # Map features to fuzzy system inputs
        self.quality_sim.input['alcohol'] = features.get('alcohol', 0)
        self.quality_sim.input['volatile_acidity'] = features.get('volatile_acidity', 0)
        self.quality_sim.input['citric_acid'] = features.get('citric_acid', 0)
        self.quality_sim.input['residual_sugar'] = features.get('residual_sugar', 0)
        self.quality_sim.input['chlorides'] = features.get('chlorides', 0)
        self.quality_sim.input['free_sulfur_dioxide'] = features.get('free_sulfur_dioxide', 0)
        self.quality_sim.input['total_sulfur_dioxide'] = features.get('total_sulfur_dioxide', 0)
        self.quality_sim.input['density'] = features.get('density', 0)
        self.quality_sim.input['ph'] = features.get('ph', 0)
        self.quality_sim.input['sulphates'] = features.get('sulphates', 0)
        
        # Compute the result
        self.quality_sim.compute()
        
        # Return predicted quality
        return self.quality_sim.output['quality']