import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = 'artifacts/preprocessor_object.pkl'
            model_path = 'artifacts/model.pkl'
            
            print('Debug: Loading preprocessor from:', preprocessor_path)
            preprocessor = load_object(preprocessor_path)
            print('Debug: Preprocessor loaded successfully')
            
            print('Debug: Loading model from:', model_path)
            model = load_object(model_path)
            print('Debug: Model loaded successfully')
            
            print('Debug: Input features shape:', features.shape)
            data_scaled = preprocessor.transform(features)
            print('Debug: Scaled data shape:', data_scaled.shape)
            
            pred = model.predict(data_scaled)
            print('Debug: Raw prediction:', pred)
            return pred
        
        except Exception as e:
            print('Debug: Error in predict:', str(e))
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, height: float, weight: float, age: int, ball_control: float, dribbling: float, 
                 slide_tackle: float, stand_tackle: float, aggression: float, reactions: float, 
                 att_position: float, interceptions: float, vision: float, composure: float, crossing: float,
                 short_pass: float, long_pass: float, acceleration: float, stamina: float, strength: float, balance: float,
                 sprint_speed: float, agility: float, jumping: float, heading: float, shot_power: float, finishing: float,
                 long_shots: float, curve: float, fk_acc: float, penalties: float, volleys: float, gk_positioning: float, gk_diving: float,
                 gk_handling: float, gk_kicking: float, gk_reflexes: float,
                 country: str, club: str):
        
        # Player Basic Info
        self.height = height
        self.weight = weight
        self.age = age
        self.country = country
        self.club = club
        
        # Technical Skills
        self.ball_control = ball_control
        self.dribbling = dribbling
        self.vision = vision
        self.composure = composure
        
        # Defensive Skills
        self.slide_tackle = slide_tackle
        self.stand_tackle = stand_tackle
        self.interceptions = interceptions
        
        # Mental Attributes
        self.aggression = aggression
        self.reactions = reactions
        self.att_position = att_position
        
        # Passing Skills
        self.crossing = crossing
        self.short_pass = short_pass
        self.long_pass = long_pass
        
        # Physical Attributes
        self.acceleration = acceleration
        self.stamina = stamina
        self.strength = strength
        self.balance = balance
        self.sprint_speed = sprint_speed
        self.agility = agility
        self.jumping = jumping
        
        # Shooting Skills
        self.heading = heading
        self.shot_power = shot_power
        self.finishing = finishing
        self.long_shots = long_shots
        self.curve = curve
        self.fk_acc = fk_acc
        self.penalties = penalties
        self.volleys = volleys
        
        # Goalkeeper Skills
        self.gk_positioning = gk_positioning
        self.gk_diving = gk_diving
        self.gk_handling = gk_handling
        self.gk_kicking = gk_kicking
        self.gk_reflexes = gk_reflexes

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'height': [self.height],
                'weight': [self.weight],
                'age': [self.age],
                'ball_control': [self.ball_control],
                'dribbling': [self.dribbling],
                'slide_tackle': [self.slide_tackle],
                'stand_tackle': [self.stand_tackle],
                'aggression': [self.aggression],
                'reactions': [self.reactions],
                'att_position': [self.att_position],
                'interceptions': [self.interceptions],
                'vision': [self.vision],
                'composure': [self.composure],
                'crossing': [self.crossing],
                'short_pass': [self.short_pass],
                'long_pass': [self.long_pass],
                'acceleration': [self.acceleration],
                'stamina': [self.stamina],
                'strength': [self.strength],
                'balance': [self.balance],
                'sprint_speed': [self.sprint_speed],
                'agility': [self.agility],
                'jumping': [self.jumping],
                'heading': [self.heading],
                'shot_power': [self.shot_power],
                'finishing': [self.finishing],
                'long_shots': [self.long_shots],
                'curve': [self.curve],
                'fk_acc': [self.fk_acc],
                'penalties': [self.penalties],
                'volleys': [self.volleys],
                'gk_positioning': [self.gk_positioning],
                'gk_diving': [self.gk_diving],
                'gk_handling': [self.gk_handling],
                'gk_kicking': [self.gk_kicking],
                'gk_reflexes': [self.gk_reflexes],
                'country': [self.country],
                'club': [self.club]
            }

            df = pd.DataFrame(custom_data_input_dict)
            
            # Ensure numeric columns are float type
            numeric_columns = ['height', 'weight', 'ball_control', 'dribbling', 
                             'slide_tackle', 'stand_tackle', 'aggression', 'reactions', 
                             'att_position', 'interceptions', 'vision', 'composure', 'crossing',
                             'short_pass', 'long_pass', 'acceleration', 'stamina', 'strength', 'balance',
                             'sprint_speed', 'agility', 'jumping', 'heading', 'shot_power', 'finishing',
                             'long_shots', 'curve', 'fk_acc', 'penalties', 'volleys', 'gk_positioning', 'gk_diving',
                             'gk_handling', 'gk_kicking', 'gk_reflexes']
            
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            # Ensure age is integer
            df['age'] = df['age'].astype(int)
            
            # Ensure categorical columns are string type
            df['country'] = df['country'].astype(str)
            df['club'] = df['club'].astype(str)
            
            print('Debug: DataFrame dtypes after conversion:')
            print(df.dtypes)
            
            return df

        except Exception as e:
            print('Debug: Error in predict:', str(e))
            raise CustomException(e, sys)
