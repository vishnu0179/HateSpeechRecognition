#rom aman_for_3_category import our_model
from hate_recog import our_model
import pandas as pd
print('begining')



hey=our_model()
print('model')
hey.get_input('i hate you')
#hey.get_input('i dont hate you')
#hey.get_input('i don\'t hate you')
#hey.get_input('i like you')
#hey.get_input('fuck you bitch')
#hey.get_input('hey what are you doing')
print('csv\n')
f= pd.read_csv(r'C:/users/amanr/desktop/labeled_data.csv')
hey.get_csv(f)

