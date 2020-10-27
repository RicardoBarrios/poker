
# coding: utf-8

# In[1]:


print('''TEXAS HOLD'EM APP TO IMPROVE SKILLS''')


# In[2]:


# Introduction : Aspects of the Game to be Programmed

#1. Different potential variables to keep into account that affect 5 decision-points: 
#     (1. Fold; 2. Check; 3. Call; 4. Bet; 5. Raise):
    
Fold = {}
Check = {}
Call = {}
Bet = {}
Raise = {}
    


# In[3]:


# Fold because hand is too weak or opponent's range is too strong

Fold[1]= 'Hand is too weak'
Fold[2]= '''Opponent's range is too strong'''
Fold


# In[4]:


# if HAND_STRENGTH < 25:
#     WEAK_HAND
    


# In[5]:


# "WEAK_HAND" variable assigned to Fold[1]

WEAK_HAND = Fold[1]
WEAK_HAND


# In[6]:


# Check because hand will improve later

Check[1]= 'Hand will improve on a later street'
Check


# In[7]:


# Call because hand is strong enough

Call[1]= "Hand is strong enough to call"
Call


# In[8]:


# Bet for value or outs 

Bet[1]= 'Bet for value'
Bet[2] = 'I have lots of outs'
Bet


# In[9]:


#  Raise for value

Raise[1]= 'Raise for value'
Raise


# In[10]:


import random
import time


# In[11]:


########### MODULE 1 : INPUTS / VARIABLES #############


# In[12]:


print('''TO DO:\n

1. APPLY PLAYER TYPES\n2. Program Quizzes 1-100\n3. APPLY RANGE CHARTS\n\n4. CREATE VARIABLE TESTING FORMULA <<<????? what does this mean?''')


# In[13]:


print('CURRENT UPDATE IN PROGRESS (Q600):')
Q600F= '''Q600Flop: AdQc top pair (Aces) in position bets medium when faced with a check due to opponent's wide range and my hand's strength'''
Q600T= 'AdQc top pair (Aces) in position checks when faced with a check when turn coordinates the board'
Q600R= 'AdQc top pair (Aces) in position bets large when faced with a check when my range is large and my hand is at the bottom of my range'

Q600F 



# In[14]:


#Quiz 600 Flop Attributes

Q600FlopAttributes = {}
Q600FlopAttributes["HAND"] = ['Ad','Qc']
Q600FlopAttributes["HAND QUALITY 1"] = 'Top Pair'
Q600FlopAttributes["POSITION"] = 'In Position'
Q600FlopAttributes["FLOP BET"] = 'Medium'
Q600FlopAttributes["FLOP BETS BEFORE YOU"] = 'Checks / No Bets'
Q600FlopAttributes["OPPONENT RANGE"] = 'Wide'


# In[15]:


#Quiz 600 Hand Rank (Card #1)

HAND = Q600FlopAttributes["HAND"]
def S(y):
    if y == 'A':
        x = 15
    elif y == 'K':
        x = 13
    elif y == 'Q':
        x = 12
    elif y == 'J':
        x = 11
    elif y == y:
        x = int(y)
    return x 
HAND_1_RANK = HAND[0][0:-1]
HAND_1_RANK
HAND_2_RANK = HAND[1][0:-1]
S(HAND_1_RANK)


# In[16]:


#Quiz 600 Hand Rank (Card #2)

S(HAND_2_RANK)


# In[17]:


#Quiz 600 Hand's Suitedness Strength

#SUITEDNESS STRENGTH
def f(x):
    if x[1][-1] == x[0][-1]:
        x = 3 #'SUITED'
    else:
        x = 0 #'UNSUITED'
    return x
SUITEDNESS_STRENGTH = f(HAND)
SUITEDNESS_STRENGTH


# In[18]:


#POCKET PAIR?
if HAND_1_RANK == HAND_2_RANK: POCKET_PAIR = 4 
else: POCKET_PAIR = 0
POCKET_PAIR


# In[19]:


#SUITED CONNECTOR?
if 3 > S(HAND_1_RANK) - S(HAND_2_RANK) > -3: CONNECTOR = 3
else: CONNECTOR = 0
CONNECTOR


# In[20]:


#HAND STRENGTH RANKED FROM 4 - 41
HAND_STRENGTH = S(HAND_1_RANK) + S(HAND_2_RANK)+ SUITEDNESS_STRENGTH + POCKET_PAIR + CONNECTOR
HAND_STRENGTH


# In[21]:


CPU1 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])
CPU2 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])
CPU3 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])
CPU4 = "|"+"  "+random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])
CPU5 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])
CPU6 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])
CPU7 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy']) 
CPU8 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])


# In[22]:


STACK1 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])
STACK2 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])
STACK3 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])
STACK4 = "|"+"  "+random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])
STACK5 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])
STACK6 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])
STACK7 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)']) 
STACK8 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])


# In[23]:


#Table
print("Q600")
print('''
                               ''' +CPU5+'''   ''' +CPU4+'''                                             
                                       _____________________________
                                    /                                \                           
                      ''' +CPU8+'''                        ''' +CPU1+'''           
                                  /                                    \ 
                                  |                                     |
                      ''' +CPU7+'''                        ''' +CPU2+'''
                                  |                                     | 
                                   \                                    /
                      ''' +CPU6+'''                        ''' +CPU3+'''
                                     \                                /
                                       ------------------------------
                                                   YOU
   ''') 


# In[24]:


#Table Player Type and Stack Sizes 
print("Q600")
print('''
                               ''' +CPU5+       '''              ''' +CPU4+'''  
                               ''' +STACK5+'''   '''+STACK4+'''
                                       _____________________________
                                    /                                \                           
                      ''' +CPU8+'''                        ''' +CPU1+'''    
                      ''' +STACK8+'''              ''' +STACK1+'''  
                                  /                                    \ 
                                  |                                     |
                      ''' +CPU7+'''                        ''' +CPU2+'''
                      ''' +STACK7+'''                ''' +STACK2+'''
                                  |                                     | 
                                   \                                    /
                      ''' +CPU6+'''                        ''' +CPU3+'''
                      ''' +STACK6+'''                 ''' +STACK3+'''
                                     \                                /
                                       ------------------------------
                                                   HERO
   ''') 


# In[25]:


#Quiz 600 in a DataFrame
print("Q600")
import pandas as pd

data = 'AdQc top pair A in position bets large when faced with a check when my range is large and my hand is at the bottom of my range'
dfAddition600 = pd.DataFrame([Q600F,Q600T,Q600R],index=["Q600F","Q600T","Q600R"])
dfAddition600


# In[26]:


#POT-ODDS DEFINITION
print('Practicing Calculating Pot-Odds'
      
      +
      '''\n\nIn poker, pot odds are the ratio of the current size of the pot to the cost of a contemplated call. Pot odds are often compared to the probability of winning a hand with a future card in order to estimate the call's expected value.''')


# In[27]:


#CALCULATION NEXT
print('In other words...')


# In[28]:


#POT-ODDS EXAMPLE
CURRENT_POT_SIZE = 200
POTENTIAL_COST_OF_CALL = 100
POT_ODDS = CURRENT_POT_SIZE / POTENTIAL_COST_OF_CALL
print('''POT_ODDS = Probability of winning a hand with a future card in order to estimate the call's expected value''')
print('Current Pot Size is '+ str(CURRENT_POT_SIZE))
print('Potential Cost of Call is '+ str(POTENTIAL_COST_OF_CALL))

print("Pot Odds are "+str(POT_ODDS)+' to 1')


# In[29]:


#FRACTION NEXT
print('In other words...')


# In[30]:


# CONVERSION FROM RATIO (2 to 1) TO PERCENTAGE/FRACTION (2/3)
print('Turn the ratio to a percent/fraction by calculating [ Risk / (Risk + Reward) ]')


# In[31]:


# NUMERATOR AKA CALL PRICE
RISK = POTENTIAL_COST_OF_CALL
RISK


# In[32]:


# PART OF DENOMINATOR IE POT SIZE PRIOR TO CALL
REWARD = CURRENT_POT_SIZE
REWARD


# In[33]:


# FRACTION/PERCENTAGE OF CALL PRICE / (POT SIZE + CALL PRICE)
RISK / (RISK + REWARD)


# In[34]:


# EXAMPLE OF POT-ODDS : BET // POT-SIZE

BET = 100
POT_SIZE = 200
X = BET / ( BET + POT_SIZE )
X


# In[35]:


# EXAMPLE OF POT-ODDS : CALL // POT-SIZE

POT_ODDS = POTENTIAL_COST_OF_CALL / (POTENTIAL_COST_OF_CALL + CURRENT_POT_SIZE)
POT_ODDS = 100 / (100+200)
POT_ODDS


# In[36]:


# POT_ODDS RATIO
POT_ODDS_RATIO = ("X TO Y ODDS")
ODDS_RATIO = str(1/POT_ODDS) + " to one"
ODDS_RATIO 


# In[37]:


# If stack size is small, bet smaller. If medium, bet medium. If larger, bet larger.

def f(x):
    if x <= 20:
        x = 'Bet Smaller'
    elif 20 < x <= 50:
        x = 'Bet Medium'
    elif 50 < x:
        x = 'Bet Larger'
    return x

MY_BET_SIZE = f(55)
    
    
MY_BET_SIZE


# In[38]:


# ALL CARDS
CARDS = ('Ah', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', 'Jh', 'Qh', 'Kh', 'Ac', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', '10c', 'Jc', 'Qc', 'Kc', 'As', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', '10s', 'Js', 'Qs', 'Ks', 'Ad', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d', 'Jd', 'Qd', 'Kd')
CARDS


# In[39]:


#SHUFFLE AND DEAL HAND AND COMMUNITY CARDS
############# NEED TO EDIT THIS A LOT FOR ACCURACY

import random
COMMUNITY_CARDS = random.sample(set(CARDS), 27)
HAND = COMMUNITY_CARDS[0:2]
#NEED TO CALCULATE OPPONPENETS HANDS IN THE DEALING
FLOP = COMMUNITY_CARDS[2:5]
BURN1 = COMMUNITY_CARDS[5]
TURN = COMMUNITY_CARDS[6]
BURN2 = COMMUNITY_CARDS[7]
RIVER = COMMUNITY_CARDS[8]
FLOP




# In[40]:


HAND = ['As','10s']
HAND


# In[41]:


FLOP = ['10c','9h','7d']
FLOP


# In[42]:


TURN = ['4h']
TURN


# In[43]:


RIVER = ['2s']
RIVER


# In[44]:


#SUITEDNESS
def f(x):
    if x[1][-1] == x[0][-1]:
        x = 'SUITED'
    else:
        x = 'UNSUITED'
    return x
SUITEDNESS = f(HAND)
SUITEDNESS


# In[45]:


#SUITEDNESS STRENGTH
def f(x):
    if x[1][-1] == x[0][-1]:
        x = 3 #'SUITED'
    else:
        x = 0 #'UNSUITED'
    return x
SUITEDNESS_STRENGTH = f(HAND)
SUITEDNESS_STRENGTH


# In[46]:


HAND_STRING_RANK = HAND[0][0:-1]+HAND[1][0:-1]
HAND_STRING_RANK


# In[47]:


HAND_1_RANK = HAND[0][0:-1]
HAND_1_RANK


# In[48]:


HAND_2_RANK = HAND[1][0:-1]
HAND_2_RANK


# In[49]:


HAND_2_RANK


# In[50]:


def S(y):
    if y == 'A':
        x = 15
    elif y == 'K':
        x = 13
    elif y == 'Q':
        x = 12
    elif y == 'J':
        x = 11
    elif y == y:
        x = int(y)
    return x 
S(HAND_1_RANK)


# In[51]:


#SECOND VARIABLE: MY HAND STRENGTH AND SUITEDNESS

HAND


# In[52]:


HAND_1_RANK = HAND[0][0:-1]
HAND_1_RANK


# In[53]:


HAND_1_RANK = S(HAND_1_RANK) 
HAND_1_RANK


# In[54]:


HAND_2_RANK = HAND[1][0:-1]
HAND_2_RANK


# In[55]:


HAND_2_RANK = S(HAND_2_RANK)
HAND_2_RANK


# In[56]:


HIGHEST_HAND_RANK = max(HAND_1_RANK,HAND_2_RANK)
HIGHEST_HAND_RANK


# In[57]:


#POCKET PAIR?
if HAND_1_RANK == HAND_2_RANK: POCKET_PAIR = 4 
else: POCKET_PAIR = 0
POCKET_PAIR


# In[58]:


#SUITED CONNECTOR?
if 3 > HAND_1_RANK - HAND_2_RANK > -3: CONNECTOR = 3
else: CONNECTOR = 0
CONNECTOR


# In[59]:


#HAND STRENGTH RANKED FROM 4 - 41
HAND_STRENGTH = S(HAND_1_RANK) + S(HAND_2_RANK)+ SUITEDNESS_STRENGTH + POCKET_PAIR + CONNECTOR
HAND_STRENGTH


# In[60]:


# #HAND STRENGTH AND SUITEDNESS
# HAND_STRENGTH_SUITEDNESS = str(HAND_STRENGTH) +' '+ SUITEDNESS
# HAND_STRENGTH_SUITEDNESS
# ^SUITEDNESS IS NOW CALCULATED INTO HAND STRENGTH


# In[61]:


#THIRD VARIABLE: MY POSITION


# In[62]:


#FOURTH VARIABLE: PLAYER TYPES
OPPONENT_TYPES = ('TAG','LAG','REG','HAPPY')
OPPONENT_TYPES


# In[63]:


#FIFTHVARIABLE: STACK SIZES
######## WORK ON SSIBB

OPPONENT_STACK_SIZE = 'shallow'

STACK_SIZE = 'SS'
STACK_SIZE_IN_BIG_BLINDS = 'SSIBB' #MY STACK SIZE IN BIG BLINDS

OPPONENT_STACK_SIZE


# In[64]:


#SIXTH VARIABLE: PRIOR BETS / POT SIZE 
ALL_PRIOR_BETS = 300


# In[65]:


POT_SIZE = ALL_PRIOR_BETS


# In[66]:


POT_SIZE


# In[67]:


CALL_AMOUNT = 200


# In[68]:


######### MODULE 1.25 : SIMULATION ############


# In[69]:


#PRE-FLOP


# In[70]:


PLAYERS_POSSIBLE = [4,5,6,7,8,9,10]
NUMBER_OF_PLAYERS = random.sample((PLAYERS_POSSIBLE), 1) #set()
NUMBER_OF_PLAYERS


# In[71]:


#HASH THISSSSSSSSSSS

NUMBER_OF_PLAYERS = [10]


# In[72]:


ACTIONS_BEFORE_YOU = list(range(1,int(NUMBER_OF_PLAYERS[0])))
ACTIONS_BEFORE_YOU


# In[73]:


TOTAL_ACTIONS = list(range(1,int(NUMBER_OF_PLAYERS[0])+1))
TOTAL_ACTIONS


# In[74]:


# (IMPORTANCE TO GAME/REQUIRED FOR GIVEN NUMBER OF PLAYERS,'POSITION',ORDER PRE-FLOP)
MAX_POSITIONS = [(4,'UTG',1),(7,'UTG+1',2),(8,'UTG+2',3),(9,'MP1',4),(10,'MP2',5),(6,'HJ',6),(5,'CO',7),(1,'BTN',8),(2,'SB',9),(3,'BB',10)]
UTG_FIRST_POSITIONS = MAX_POSITIONS
UTG_FIRST_POSITIONS


# In[75]:


POSITIONS_SORTED_BY_IMPORTANCE = sorted(MAX_POSITIONS, key=lambda tup: tup[0])
POSITIONS_SORTED_BY_IMPORTANCE


# In[76]:


############NEED THIS TO CALCULATE OPPONENT DATAFRAME
ALL_POSITIONS_IN_THIS_GAME = POSITIONS_SORTED_BY_IMPORTANCE[0:TOTAL_ACTIONS[-1]]
ALL_POSITIONS_IN_THIS_GAME


# In[77]:


STACK_SIZES_POSSIBLE = ['Stack is <50 Big Blinds',
                       'Stack is >50 Big Blinds <150',
                       'Stack is >150 Big Blinds']
STACK_SIZES_POSSIBLE


# In[78]:


########NEED THIS TO CALCULATE OPPONENT DATAFRAME

from random import choices

ALL_STACK_SIZES = choices(STACK_SIZES_POSSIBLE,
                          k=NUMBER_OF_PLAYERS[0])
ALL_STACK_SIZES


# In[79]:


POSITIONS_STACKS_FLOP = set(zip(ALL_POSITIONS_IN_THIS_GAME,ALL_STACK_SIZES))
POSITIONS_STACKS_FLOP


# In[80]:


def Convert(tup, di): 
    for a, b in tup: 
        di.setdefault(a, []).append(b) 
    return di

dictionary = {}
print (Convert(POSITIONS_STACKS_FLOP, dictionary))


# In[81]:


UTG_FIRST_POSITIONS


# In[82]:


UTG_FIRST_POSITIONS_STACKS_PREFLOP = set(zip(UTG_FIRST_POSITIONS , ALL_STACK_SIZES))
UTG_FIRST_POSITIONS_STACKS_PREFLOP


# In[83]:


OPPONENT_ACTIONS_POSSIBLE = ['fold','check','call',
                    'bet small',
                    'bet medium',
                    'bet large']
                   #'YOUR MOVE']
OPPONENT_ACTIONS_POSSIBLE


# In[84]:


OPTIONAL_MOVES_PREVIOUS_POSSIBLE_MOVES = {
 'zero':[0],
 'fold':['bet small','call','bet medium','bet large'],
 'check':['StreetBegins','check'],
 'bet small':['StreetBegins','check','fold'],
 'call':['bet small','bet medium','bet large','fold'],
 'bet medium':['StreetBegins','fold','check','call'],
 'bet large':['StreetBegins','fold','check','call','bet small']
}

OPTIONAL_MOVES_PREVIOUS_POSSIBLE_MOVES


# In[85]:


FOLD_GIVEN_PREVIOUS_MOVES = {'fold':['bet small','call','bet medium','bet large']}
CHECK_GIVEN_PREVIOUS_MOVES = {'check':['StreetBegins','check']}
BET_SMALL_GIVEN_PREVIOUS_MOVES = {'bet small':['StreetBegins','check','fold'],}
CALL_GIVEN_PREVIOUS_MOVES = {'call':['bet small','bet medium','bet large','fold'],}
BET_MEDIUM_GIVEN_PREVIOUS_MOVES = {'bet medium':['StreetBegins','fold','check','call'],}
BET_LARGE_GIVEN_PREVIOUS_MOVES = {'bet large':['StreetBegins','fold','check','call','bet small']}
ALL_IN_GIVEN_PREVIOUS_MOVES = {'all-in':['StreetBegins','fold','check','call','bet small']}

CHECK_GIVEN_PREVIOUS_MOVES


# In[86]:


Dictionary_Of_Possible_Bets = {
'StreetBegins' : ['Check','Bet Small','Bet Medium','Bet Large','All-In','Check'],
'Check': ['Fold','Check','Bet Small','Bet Medium','Bet Large','All-In'],                           
'Bet Small' : ['Fold','Call','Bet Medium','Bet Large','All-In','Call'],
'Call' : ['Fold','Call','Bet Medium','Bet Large','All-In','Call'],
'Bet Medium' : ['Fold','Call','Bet Large','All-In','Call','Fold'],
'Bet Large' : ['Fold','Call','Call','All-In','Call','Fold'],
'Fold' : ['Fold','Call','Bet Small','Bet Medium','Bet Large','All-In'],
'All-In' : ['Fold','Call','Fold','Call','Fold','All-In']
                                }
Dictionary_Of_Possible_Bets
                               


# In[87]:


dfPossibleBets = pd.DataFrame.from_dict(Dictionary_Of_Possible_Bets,orient='index')
dfPossibleBets.fillna(value=pd.np.nan, inplace=True)
dfPossibleBets.columns = ['Possibility 1','Possibility 2','Possibility 3','Possibility 4','Possibility 5','Possibility 6']
# dfPossibleBets['Random Selections'] = "Placeholder"
dfPossibleBets


# In[88]:


# dfPossibleBets['Impossible Previous Bet'] = ['Fold', 'ALL BETS POSSIBLE AFTER CHECK', '']


# In[89]:


dfPossibleBets.loc['Call']


# In[90]:


dfPossibleBets.values


# In[91]:


import numpy as np
random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections


# In[92]:


random_selections_list = random_selections.tolist()
random_selections_list


# In[93]:


dfPossibleBets['Random Selections 1'] = random_selections_list
dfRandomSelections = dfPossibleBets.replace("Placeholder","NaN")
dfRandomSelections


# In[94]:


BET0 = 'StreetBegins'
BET0


# In[95]:


BET1 = dfRandomSelections['Random Selections 1'].loc[BET0]
dfRandomSelections['Bet #1'] = BET1
BET1


# In[96]:


random_selections_list_2 = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values]).tolist()# if str(i) != 'NaN'])
dfRandomSelections['Random Selections 2'] = random_selections_list_2
dfRandomSelections


# In[97]:


BET2 = dfRandomSelections['Random Selections 2'].loc[BET1]
dfRandomSelections['Bet #2'] = BET2
BET2


# In[98]:


random_selections_list_3 = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values]).tolist()# if str(i) != 'NaN'])
dfRandomSelections['Random Selections 3'] = random_selections_list_3
dfRandomSelections


# In[99]:


BET3 = dfRandomSelections['Random Selections 3'].loc[BET2]
dfRandomSelections['Bet #3'] = BET3
BET3


# In[100]:


##########EVENTUALLY CREATE FOR LOOP TO ADD RANDOM SELECTION COLUMN
random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_4 = random_selections.tolist()
dfRandomSelections['Random Selections 4'] = random_selections_list_4
BET4 = dfRandomSelections['Random Selections 4'].loc[BET3]
dfRandomSelections['Bet #4'] = BET4

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_5 = random_selections.tolist()
dfRandomSelections['Random Selections 5'] = random_selections_list_5
BET5 = dfRandomSelections['Random Selections 5'].loc[BET4]
dfRandomSelections['Bet #5'] = BET5

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_6 = random_selections.tolist()
dfRandomSelections['Random Selections 6'] = random_selections_list_6
BET6 = dfRandomSelections['Random Selections 6'].loc[BET5]
dfRandomSelections['Bet #6'] = BET6

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_7 = random_selections.tolist()
dfRandomSelections['Random Selections 7'] = random_selections_list_7
BET7 = dfRandomSelections['Random Selections 7'].loc[BET6]
dfRandomSelections['Bet #7'] = BET7

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_8 = random_selections.tolist()
dfRandomSelections['Random Selections 8'] = random_selections_list_8
BET8 = dfRandomSelections['Random Selections 8'].loc[BET7]
dfRandomSelections['Bet #8'] = BET8

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_9 = random_selections.tolist()
dfRandomSelections['Random Selections 9'] = random_selections_list_9
BET9 = dfRandomSelections['Random Selections 9'].loc[BET8]
dfRandomSelections['Bet #9'] = BET9

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_10 = random_selections.tolist()
dfRandomSelections['Random Selections 10'] = random_selections_list_10
BET10 = dfRandomSelections['Random Selections 10'].loc[BET9]
dfRandomSelections['Bet #10'] = BET10

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_11 = random_selections.tolist()
dfRandomSelections['Random Selections 11'] = random_selections_list_11
BET11 = dfRandomSelections['Random Selections 11'].loc[BET10]
dfRandomSelections['Bet #11'] = BET11

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_12 = random_selections.tolist()
dfRandomSelections['Random Selections 12'] = random_selections_list_12
BET12 = dfRandomSelections['Random Selections 12'].loc[BET11]
dfRandomSelections['Bet #12'] = BET12

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_13 = random_selections.tolist()
dfRandomSelections['Random Selections 13'] = random_selections_list_13
BET13 = dfRandomSelections['Random Selections 13'].loc[BET12]
dfRandomSelections['Bet #13'] = BET13

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_14 = random_selections.tolist()
dfRandomSelections['Random Selections 14'] = random_selections_list_14
BET14 = dfRandomSelections['Random Selections 14'].loc[BET13]
dfRandomSelections['Bet #14'] = BET14

random_selections = pd.Series([np.random.choice(i,1)[0] for i in dfPossibleBets.values])# if str(i) != 'NaN'])
random_selections_list_15 = random_selections.tolist()
dfRandomSelections['Random Selections 15'] = random_selections_list_15
BET15 = dfRandomSelections['Random Selections 15'].loc[BET14]
dfRandomSelections['Bet #15'] = BET15

dfRandomSelections


# In[101]:


BETS_ROUND_ONE = [BET1, BET2, BET3, BET4, BET5, BET6, BET7, BET8, BET9, BET10]
BETS_ROUND_ONE
# BETS_PERCENTAGE_OF_STACK = BETS_ROUND_ONE.replace('Bet Medium',.5)
# BETS_PERCENTAGE_OF_STACK


# In[102]:


BETS_ROUND_ONE


# In[103]:


BETS_ROUND_ONE = ['Bet Medium / 25%' if x=='Bet Medium' else x for x in BETS_ROUND_ONE]
BETS_ROUND_ONE = ['Bet Small / 10%' if x=='Bet Small' else x for x in BETS_ROUND_ONE]
BETS_ROUND_ONE = ['Bet Large / 40%' if x=='Bet Large' else x for x in BETS_ROUND_ONE]
BETS_ROUND_ONE = ['Fold / Bet 0%' if x=='Fold' else x for x in BETS_ROUND_ONE]
BETS_ROUND_ONE = ['Go All-In / Bet 100%' if x=='All-In' else x for x in BETS_ROUND_ONE]
BETS_ROUND_ONE = ['Call / Bet Same as Opponent' if x=='Call' else x for x in BETS_ROUND_ONE]

BETS_ROUND_ONE


# In[104]:


# import itertools
# for a, b in itertools.combinations(mylist, 2):
#     compare(a, b)


# In[105]:


# for m, x in BETS_ROUND_ONE:
#     if BETS_ROUND_ONE[x] < BETS_ROUND_ONE[m*x]:
#         BETS_ROUND_ONE[x] = 'Call'
# BETS_ROUND_ONE


# In[106]:


ALL_STACK_SIZES = choices(STACK_SIZES_POSSIBLE,
                          k=NUMBER_OF_PLAYERS[0])
ALL_STACK_SIZES





# In[107]:


ALL_POSITIONS_IN_THIS_GAME = ALL_POSITIONS_IN_THIS_GAME[3:]+ALL_POSITIONS_IN_THIS_GAME[0:3]
ALL_POSITIONS_IN_THIS_GAME


# In[108]:


def merge(UTG_FIRST_POSITIONS, ALL_STACK_SIZES, BETS_ROUND_ONE):
    merged_list = [(UTG_FIRST_POSITIONS[i], ALL_STACK_SIZES[i], BETS_ROUND_ONE[i]) for i in range(0, len(UTG_FIRST_POSITIONS))]
    return merged_list


# In[109]:


FIRST_ROUND_OF_BETTING = merge(UTG_FIRST_POSITIONS,ALL_STACK_SIZES,BETS_ROUND_ONE)
FIRST_ROUND_OF_BETTING


# In[110]:


# BetLargeDictionary = {'Bet Large': ['Fold', 'Call,', 'Bet Large', 'All-In']}
# BetMediumDictionary = {'Bet Medium': ['Fold', 'Bet Large', 'All-In']}
# BetSmallDictionary = {'Bet Small': ['Fold', 'Call', 'Bet Medium', 'Bet Large', 'All-In']}
# CallDictionary = {'Call' : ['Fold', 'Bet Medium', 'Bet Large', 'All-In']}
# CheckDictionary = {'Check' : ['Fold', 'Check', 'Bet Small', 'Bet Medium', 'Bet Large', 'All-In']}
# PRIOR_BET_Dictionary = {'PRIOR_BET' : ['ALL POSSIBLE BETS AFTER PRIOR BET']}
# StreetBeginsDictionary = {'Street Begins' : ['Check', 'Bet Small', 'Bet Medium', 'Bet Large', 'All-In']}

# CallDictionary


# In[111]:


######## NEED TO ELIMINATE PLAYERS ONCE THEY FOLD
######## CANNOT ALLOW CHECKINGS AFTER A BET


# In[112]:




#^^^^^^^^^^RANDOMIZED
########################################################################################################################
########################################################################################################################
########################################################################################################################
#DESIGNED:

DESIGNED_POSITIONS_STACKS_SORTED_ACTIONS = '''[(((4, 'UTG', 1), 'Stack is >50 Big Blinds <150'), 'initial bet or min-raise'),
 (((7, 'UTG+1', 2), 'Stack is >50 Big Blinds <150'), 'fold'),
 (((8, 'UTG+2', 3), 'Stack is >150 Big Blinds'), 'check'),
 (((9, 'MP1', 4), 'Stack is <50 Big Blinds'),
  'bet large (initial bet or raise/re-raise)'),
 (((10, 'MP2', 5), 'Stack is >50 Big Blinds <150'),
  'bet large (initial bet or raise/re-raise)'),
 (((6, 'HJ', 6), 'Stack is >50 Big Blinds <150'),
  'bet small (initial bet or min-raise)'),
 (((5, 'CO', 7), 'Stack is <50 Big Blinds'), 'fold'),
 (((1, 'BTN', 8), 'Stack is >150 Big Blinds'), 'check'),
 (((2, 'SB', 9), 'Stack is >50 Big Blinds <150'), 'fold'),
 (((3, 'BB', 10), 'Stack is >150 Big Blinds'),
  'YOUR MOVE')]'''

DESIGNED_POSITIONS_STACKS_SORTED_ACTIONS


# In[113]:


######### MODULE 2 : DECISIONS ############
######### PART ONE : PRE-FLOP #############


# In[114]:


#*** means that I need to come back to this
RAISE_REASONS = ('hand makes nut flush', #***need to find source video- not sure if given flop or not
                 'hand makes marginal top pair', #DONE
                 'opponent doesnt see this coming', #***LEVEL5
                 'hand works well as a bluff since it has a good blocker', #DONE
                 'good post-flop playability', #***need to determine this range LEVEL3
                 'my range is better than the opponents') #***need to determine how to figure this out LEVEL4


# In[115]:


RAISE_REASONS


# In[116]:


THREE_BET_REASONS = ('opponent folds a lot of their range') #***LEVEL5


# In[117]:


CHECK_RAISE_REASONS = ('')


# In[118]:


OPTIONS = ('bet small','bet medium','bet large','all-in','check','fold','call')


# In[119]:


OPTIONS


# In[120]:


LIMP = ['Everyone folded to me in the small blind and Im doing this with a large amount of my range'] #DONE


# In[121]:


BET_SMALL_REASONS = ['100% continuation bet strategy', #***LEVEL4
                     'Im on the button and the hand is good enough to bet (Q8o)'] #DONE


# In[122]:


# POSITION = MY_POSITION[1]
# POSITION


# In[123]:


COMMUNITY_CARDS


# In[124]:


HAND


# In[125]:


FLOP


# In[126]:


HAND_1_RANK


# In[127]:


HAND_2_RANK


# In[128]:


FLOP_1_RANK = S(FLOP[0][0:-1])
FLOP_1_RANK


# In[129]:


FLOP_2_RANK = S(FLOP[1][0:-1])
FLOP_2_RANK


# In[130]:


FLOP_3_RANK = S(FLOP[2][0:-1])
FLOP_3_RANK


# In[131]:


TURN


# In[132]:


TURN_RANK = TURN[0][0:-1]
TURN_RANK


# In[133]:


TURN_RANK = S(TURN_RANK)
TURN_RANK


# In[134]:


RIVER


# In[135]:


RIVER_RANK = RIVER[0][0:-1]
RIVER_RANK


# In[136]:


RIVER_RANK = S(RIVER_RANK)
RIVER_RANK


# In[137]:


#RIVER TOP TWO PAIR - GO ALL IN


# In[138]:


#hand makes marginal top pair if highest FLOP RANK is 11 and one HAND RANK is equal to highest FLOP RANK

HIGHEST_FLOP_RANK = max(FLOP_1_RANK,FLOP_2_RANK,FLOP_3_RANK)
HIGHEST_FLOP_RANK


# In[139]:


HIGHEST_HAND_RANK = max(HAND_1_RANK,HAND_2_RANK)
HIGHEST_HAND_RANK


# In[140]:


FLOP_1_RANK


# In[141]:


POSITIONS_STACKS_FLOP


# In[142]:


FIRST_ROUND_OF_BETTING


# In[143]:


CPU1_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[0][1:3])
CPU2_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[1][1:3])
CPU3_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[2][1:3])
CPU4_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[3][1:3])
CPU5_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[4][1:3])
CPU6_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[5][1:3])
CPU7_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[6][1:3])
CPU8_SITUATION = FIRST_ROUND_OF_BETTING[0][0][1] + ' ' + str(FIRST_ROUND_OF_BETTING[7][1:3])

CPU7_SITUATION


# In[144]:


My_Situation = random.choice(FIRST_ROUND_OF_BETTING)
My_Situation


# In[145]:


MY_TURN_NUMBER = FIRST_ROUND_OF_BETTING.index(My_Situation)
MY_TURN_NUMBER


# In[146]:


My_Position=My_Situation[0][1]
My_Position


# In[147]:


[item for item in FIRST_ROUND_OF_BETTING if My_Situation in item]


# In[148]:


FIRST_ROUND_OF_BETTING


# In[149]:


BETS_ROUND_ONE


# In[150]:


BETTING_TO_YOU = BETS_ROUND_ONE[0:MY_TURN_NUMBER]
BETTING_TO_YOU


# In[151]:


import re
TOTAL_BETS_PRIOR_TO_YOU = sum(int(x) for x in re.findall(r'[0-9]+', "".join(list(BETTING_TO_YOU))))
TOTAL_BETS_PRIOR_TO_YOU


# In[152]:


My_Position


# In[153]:


COST_OF_THE_CALL = BETTING_TO_YOU[-1]
COST_OF_THE_CALL


# In[154]:


# NUMBER_COST_OF_THE_CALL = re.sub('[^0-9]', '', COST_OF_THE_CALL)
# NUMBER_COST_OF_THE_CALL


# In[155]:


# SIZE_OF_THE_POT = int(TOTAL_BETS_PRIOR_TO_YOU)
# SIZE_OF_THE_POT


# In[156]:


# POT_ODDS RATIO
# POT_ODDS_RATIO_2 = ("X TO Y ODDS")
# ODDS_RATIO_2 = str(1/POT_ODDS) + " to one"
# ODDS_RATIO_2 


# In[157]:


import pandas as pd
data = pd.DataFrame({"HAND": HAND},{"HAND_STRENGTH":HAND_STRENGTH},{"POSITION":My_Position})


# In[158]:


data


# In[159]:


data['POT ODDS'] = POT_ODDS
data


# In[160]:


FIRST_ROUND_OF_BETTING


# In[161]:


#import this (sic)


# In[162]:


Position_list = [(4,'UTG',1),(7,'UTG+1',2),(8,'UTG+2',3),(9,'MP1',4),(10,'MP2',5),(6,'HJ',6),(5,'CO',7),(1,'BTN',8),(2,'SB',9),(3,'BB',10)]
Position_list


# In[163]:


#CAll PREFLOP SB 3.5 TO 1 POT ODDS BECAUSE THERE'S A $2 $2 $2 bet treain in 
# front of you and you only need $1 to call, hand can get top pair sometimes

# dfPreflop.loc(0)['REASONS TO CALL'] = dfPreflop['']
# dfPreflop.at["Preflop","REASONS TO CALL"] = FLOP

# dfPreflop['REASONS TO CALL'] = 'str' + dfPreflop['REASONS TO CALL'].astype(str)
# dfPreflop

# dfPreflop['REASONS TO BET CALL'] = dfPreflop['POT ODDS'].apply(lambda x: 
#     '''Call Preflop when you have 3.5 to 1 pot odds after 3 previous same-size bets and one caller behind, especially in the Small Blind''' 
#     #########FIRST TO ACT INCLUDING FOLDS:
#         if POT_ODDS > 3 
#             and HAND_STRENGTH >= 30
#                 and My_Position == 'SB'
#                         and x == 'UTG'
#                             and HAND_STRENGTH >= 18
#                                 else np.NaN)


# In[164]:


# dfPreflop


# In[165]:


newlist=random.shuffle(FIRST_ROUND_OF_BETTING)
print(newlist)


# In[166]:


dfMyHand = pd.DataFrame(columns = ["CARDS","HAND STRENGTH","MY POSITION"], index = ['Preflop','Flop','Turn','River'])
dfMyHand


# In[167]:


dfMyHand.at['Preflop','CARDS'] = HAND
dfMyHand['HAND STRENGTH']=HAND_STRENGTH
dfMyHand['MY POSITION']=My_Position
dfMyHand['MY STACK SIZE']=My_Situation[1]
dfMyHand['HIGHEST HAND RANK']=HIGHEST_HAND_RANK
dfMyHand['POT ODDS'] = POT_ODDS
dfMyHand


# In[168]:


dfPreflop = dfMyHand


# In[169]:


dfFlop = dfMyHand
# dfFlop['FLOP']=str(FLOP)
# dfFlop['HIGHEST FLOP RANK']=HIGHEST_FLOP_RANK
dfFlop


# In[170]:


import numpy as np


# In[171]:


###############MANUALLY ENTERED CRITERIA
dfPreflop['HAND STRENGTH'] = 20
dfPreflop['MY POSITION'] = 'BTN'
dfPreflop


# In[172]:


dfPreflop['REASONS TO LIMP/BET SMALL'] = dfPreflop['HAND STRENGTH'].apply(lambda x: '''I'm on the button and the hand is good enough to bet (Q8o [strength 20] or better)''' 
    if 28 > x > 19 and 'MY POSITION' == 'BTN'else np.NaN)

#'''I'm on the button but the hand is not good enough to bet (Q8o [strength 20] or worse)''')                 

dfPreflop


# In[173]:


dfPreflop['REASONS TO CALL'] = dfPreflop['HAND STRENGTH'].apply(lambda x: '''TESTING''' 
    if 28 > x > 19 and 'MY POSITION' == 'BTN'else np.NaN)

#'''I'm on the button but the hand is not good enough to bet (Q8o [strength 20] or worse)''')                 

dfPreflop


# In[174]:


pd.set_option('max_colwidth',800)
dfPreflop


# In[175]:


BET_MEDIUM_REASONS = ['UTG raised, then it folded around to me and Im doing this with a large amount of my range']


# In[176]:


BET_LARGE_REASONS = ['medium-strength hands identified as such to opponents (small raise) require a larger raise', #Quiz 259: 44suited raise 4x BB
                    'bet 3x big blind when deep stacked with a good hand first to act'] #Q264 88 suited preflop first to act


# In[177]:


ALL_IN_REASONS = ['get more value from strong made hands']


# In[178]:


CHECK_REASONS = ['would have previously checked my range that now improves',
                 'marginal made hand with an overcard kicker that makes sense to check '#Q262 Flop
                ]


# In[179]:


FOLD_REASONS = ['not many good cards to come that will help my hand']


# In[180]:


CALL_REASONS = ['getting good pot-odds to call',
                'play the entire hand in-position',
                'hand is too strong to fold',
                'getting the right price to call',
               'my hand is near the top of my range'#Q262-Turn
               ]


# In[181]:


VIBES = ['that he has pocket 7s rather than a made hand']


# In[182]:


CARDS = ('Ah', '2h', '3h', '4h', '5h', '6h', '7h', '8h', '9h', '10h', 'Jh', 'Qh', 'Kh', 'Ac', '2c', '3c', '4c', '5c', '6c', '7c', '8c', '9c', '10c', 'Jc', 'Qc', 'Kc', 'As', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', '10s', 'Js', 'Qs', 'Ks', 'Ad', '2d', '3d', '4d', '5d', '6d', '7d', '8d', '9d', '10d', 'Jd', 'Qd', 'Kd')


# In[183]:


CARDS


# In[184]:


# SIMPLIFIED_ACTIONS_THROUGH_TO_YOU = [word.replace('bet large (initial bet or raise/re-raise)','bet large') for word in ACTIONS_THROUGH_TO_YOU]
# SIMPLIFIED_ACTIONS_THROUGH_TO_YOU = [word.replace('bet medium (initial bet or raise/re-raise)','bet medium') for word in SIMPLIFIED_ACTIONS_THROUGH_TO_YOU]
# SIMPLIFIED_ACTIONS_THROUGH_TO_YOU = [word.replace('bet small (initial bet or min-raise)','bet small') for word in SIMPLIFIED_ACTIONS_THROUGH_TO_YOU]

# SIMPLIFIED_ACTIONS_THROUGH_TO_YOU


# In[185]:


#FOLD TO YOU PREFLOP

# FOLD_TO_YOU = ACTIONS_THROUGH_TO_YOU

# for n, i in enumerate(FOLD_TO_YOU):
#     if i != 'YOUR MOVE':
#         FOLD_TO_YOU[n] = 'fold'

# FOLD_TO_YOU


# In[186]:


#UTG BET THEN FOLD TO YOU PREFLOP

# UTG_BET_FOLD_TO_YOU = ACTIONS_THROUGH_TO_YOU#SIMPLIFIED_ACTIONS_THROUGH_TO_YOU

# for n, i in enumerate(UTG_BET_FOLD_TO_YOU):
#     if i != 'YOUR MOVE':
#         UTG_BET_FOLD_TO_YOU[n] = 'fold'

# UTG_BET_FOLD_TO_YOU[0] = 'bet'
# UTG_BET_FOLD_TO_YOU


# In[187]:


# n = 1
# ACTIONS_THROUGH_TO_YOU = [x[n] for x in POSITIONS_STACKS_ACTIONS_THROUGH_TO_YOU ]
# ACTIONS_THROUGH_TO_YOU


# In[188]:


#ACTIONS_THROUGH_TO_YOU_'BET'


# In[189]:


# FOLD_TO_YOU_DATAFRAME = pd.DataFrame({'Action': FOLD_TO_YOU})
# FOLD_TO_YOU_DATAFRAME


# In[190]:


# ACTUAL_ACTIONS_DATAFRAME = pd.DataFrame({'Action': ACTIONS_THROUGH_TO_YOU})#SIMPLIFIED_ACTIONS_THROUGH_TO_YOU})
# ACTUAL_ACTIONS_DATAFRAME


# In[191]:


# ACTUAL_ACTIONS_ARRAY = ACTUAL_ACTIONS_DATAFRAME['Action'].unique()
# ACTUAL_ACTIONS_ARRAY


# In[192]:


# # SIMPLIFIED_ACTIONS_THROUGH_TO_YOU_ARRAY = np.array(SIMPLIFIED_ACTIONS_THROUGH_TO_YOU, dtype=object)
# # SIMPLIFIED_ACTIONS_THROUGH_TO_YOU_ARRAY
# ACTIONS_THROUGH_TO_YOU_ARRAY = np.array(ACTIONS_THROUGH_TO_YOU, dtype= object)
# ACTIONS_THROUGH_TO_YOU


# In[193]:


# #Are all actions prior to you folds?
# FOLD_TO_YOU = FOLD_TO_YOU_DATAFRAME['Action'].unique()
# FOLD_TO_YOU


# In[194]:


import numpy as np
FOLD_ARRAY = np.array(['fold', 'YOUR MOVE'], dtype=object)
FOLD_ARRAY


# In[195]:


# np.array_equal(FOLD_TO_YOU,FOLD_ARRAY)


# In[196]:


if HAND_STRENGTH < 25:
    dfPreflop['REASONS TO FOLD'] = 'Hand is too weak to call'
else: np.NaN


# In[197]:


#Limp if everyone folded to me in the small blind because I'm doing this with a large amount of my range: All previous actions are FOLD and my position is SB pre-flop

dfPreflop['REASONS TO LIMP/BET SMALL'] = dfPreflop['MY POSITION'].apply(lambda x: '''Everyone folded to me in the small blind and I'm limping with a large amount of my range''' if x == 'SB'
    and np.array_equal(FOLD_TO_YOU,ACTUAL_ACTIONS_ARRAY) == True
    else np.NaN)
dfPreflop


# In[198]:


FLOP_2_RANK


# In[199]:


FLOP_3_RANK


# In[200]:


HAND_RANKS_ONLY = [HAND_1_RANK, HAND_2_RANK]
HAND_RANKS_ONLY


# In[201]:


FLOP_RANKS_ONLY = [FLOP_1_RANK, FLOP_2_RANK, FLOP_3_RANK]
FLOP_RANKS_ONLY


# In[202]:


TURN_RANK


# In[203]:


HAND = ['As', '2s']
HAND


# In[204]:


HAND_STRENGTH


# In[205]:


dfPreflop


# In[206]:


PAIR = [x for x in HAND_RANKS_ONLY if x in FLOP_RANKS_ONLY]
PAIR


# In[207]:


PAIR


# In[208]:


# PAIR[0]


# In[209]:


#if there is a number in PAIR, PAIR_STRENGTH = 4. NOT SURE HOW THIS CODE IS WORKING
# if 1 in PAIR: PAIR_STRENGTH = 4
# else: PAIR_STRENTGH = 0
# PAIR_STRENGTH


# In[210]:


FLOP


# In[211]:


BURN1


# In[212]:


TURN


# In[213]:


BURN2


# In[214]:


RIVER


# In[215]:


HAND_STRENGTH


# In[216]:


dfPreflop


# In[217]:


dfPreflop['REASONS TO RAISE'] = dfPreflop['HAND STRENGTH'].apply(lambda x: '''Hand makes marginal top pair''' 
    if x == HIGHEST_HAND_RANK and 8 < x < 12 else np.NaN)

#'''I'm on the button but the hand is not good enough to bet (Q8o [strength 20] or better)''')                 

dfPreflop


# In[218]:


HAND_STRENGTH


# In[219]:


dfPreflop


# In[220]:


#Hand works well as a bluff since it has good blockers if HAND 1 or HAND 2 has RANK 15

dfPreflop['REASONS TO RAISE'] = dfPreflop['HIGHEST HAND RANK'].apply(lambda x: '''Hand works well as a bluff since it has a good blocker'''
    if x == 15 and HAND_STRENGTH < 30 else np.NaN)

dfPreflop


# In[221]:


# SIMPLIFIED_ACTIONS_THROUGH_TO_YOU


# In[222]:


# UTG_BET_FOLD_ARRAY = np.array([UTG_BET_FOLD_TO_YOU[0].startswith('bet'),'fold', 'YOUR MOVE'], dtype=object)
# UTG_BET_FOLD_ARRAY


# In[223]:


# UTG_BET_FOLD_ARRAY_TO_BET = np.array([True,'fold','YOUR MOVE'], dtype=object)
# UTG_BET_FOLD_ARRAY_TO_BET


# In[224]:


#FOLDS TO YOU / FIRST TO ACT


# In[225]:


####CREATE 'POT ODDS' CALCULAOR/GAME#########################


# In[226]:


dfPreflop


# In[227]:


# dfPreflop['ACTIONS'].loc= SIMPLIFIED_ACTIONS_THROUGH_TO_YOU
# dfPreflop


# In[228]:


# dfBetsPreflop = dfPreflop
# dfBetsPreflop[POSITIONS_STACKS_ACTIONS_THROUGH_TO_YOU] = 
# dfBetsPreflop


# In[229]:


# BET_MEDIUM_REASONS = ['UTG raised (ACTIONS_THROUGH_TO_YOU contains "bet"), then it folded around to me (and Im doing this with a large amount of my range']

dfPreflop['REASONS TO BET MEDIUM'] = dfPreflop['MY POSITION'].apply(lambda x: '''Someone raised and then everyone folded to me in the small blind and I'm doing this with a large amount of my range''' if x == 'SB'
    and np.array_equal(UTG_BET_FOLD_ARRAY_TO_BET,UTG_BET_FOLD_ARRAY) == True
    else np.NaN)
dfPreflop


# In[230]:


dfPreflop.at["Flop","CARDS"] = FLOP
dfFlop = dfPreflop
dfFlop


# In[231]:


dfFlop.at["Turn","CARDS"] = [TURN]
dfTurn = dfFlop
dfTurn


# In[232]:


dfTurn.at["River","CARDS"] = [RIVER]
dfRiver = dfTurn
dfRiver


# In[233]:


TOP_7_CARDS = dfRiver.CARDS.tolist()
TOP_7_CARDS


# In[234]:


YOUR_CARDS_PLUS_FLOP = TOP_7_CARDS[0:2]
YOUR_CARDS_PLUS_FLOP


# In[235]:


YOUR_CARDS = dfRiver.CARDS.tolist()[0]
YOUR_CARDS


# In[236]:


FLOP


# In[237]:


List_of_your_cards_plus_flop = YOUR_CARDS + FLOP
List_of_your_cards_plus_flop


# In[238]:


import string
listA = [''.join(x for x in par if x not in string.punctuation) for par in List_of_your_cards_plus_flop]
listB = ''.join(listA)
listB


# In[239]:


Card_Numbers = listB[::2]
Card_Numbers


# In[240]:


Card_Suits_Plus_Z = 'Z'+listB
Card_Suits = Card_Suits_Plus_Z[::2]
Card_Suits_Minus_Z = Card_Suits[1:]
Card_Suits_Minus_Z


# In[241]:


#count number of suits in hand plus flop
import collections
results = collections.Counter(Card_Suits_Minus_Z)
results


# In[242]:


Backdoor_Flush_Draw = [id for id, count in results.items() if count == 3]
Backdoor_Flush_Draw


# In[243]:


dfRiver


# In[244]:


#count number of numbers in hand plus flop
import collections
results = collections.Counter(Card_Numbers)
results


# In[245]:


Pair_the_Flop = [id for id, count in results.items() if count > 1]
Pair_the_Flop


# In[246]:


# STRENGTH_OF_YOUR_CARDS_PLUS_FLOP
######## CARD NUMBER STRENGTH >>>>>>>>>MAYBE DO THIS LATER, SEEMS UNNECESSARY AFTER THE FLOP
######## PAIR STRENGTH
######## 3 OF A KIND STRENGTH
######## STRAIGHT STRENGTH
######## FLUSH STRENGTH
######## FOUR OF A KIND STRENGTH
######## STRAIGHT FLUSH STRENGTH
######## ROYAL FLUSH STRENGTH


# In[247]:


HAND_STRENGTH = 30


# In[248]:


My_Position


# In[249]:


MY_STACK_SIZE = My_Situation[1]
MY_STACK_SIZE


# In[250]:


#Bet 3x big blind when deep stacked with a good hand first to act'] #Q264 88 suited preflop first to act
dfPreflop['REASONS TO BET LARGE'] = dfPreflop['MY POSITION'].apply(lambda x: 
    '''Bet 3x big blind when deep stacked with a good hand first to act''' 
    #########FIRST TO ACT INCLUDING FOLDS:
        if MY_STACK_SIZE == "Stack is >150 Big Blinds" 
            and HAND_STRENGTH >= 30
                and My_Position == 'UTG'
                    or MY_STACK_SIZE == "Stack is >150 Big Blinds" 
                        and x == 'UTG'
                            and HAND_STRENGTH >= 30
                                else np.NaN)
    ##########FIRST TO ACT INCLUDING FOLDS^^:
dfPreflop


# In[251]:


HAND_RANKS_ONLY


# In[252]:


FLOP_RANKS_ONLY


# In[253]:


OVERCARDS = [max(i) for i in zip(FLOP_RANKS_ONLY,HAND_RANKS_ONLY)]
OVERCARD = max(OVERCARDS)
OVERCARD


# In[254]:


OVERCARD_IN_HAND = OVERCARD in HAND_RANKS_ONLY
OVERCARD_IN_HAND


# In[255]:


# Check marginal made hand (medium pair [6s to 10s]) with an overcard kicker #Q262 Flop

try:
    # the code that can cause the error
    if 11 > PAIR[0] > 5 and OVERCARD_IN_HAND == True: 
        PAIR_WITH_OVERCARD = 'Marginal made hand with overcard kicker'

        
except IndexError: # catch the error
    pass




# In[256]:


try:
    if PAIR_WITH_OVERCARD == 'Marginal made hand with overcard kicker':
        dfPreflop['REASONS TO CHECK'] = 'Check marginal made hand (medium pair [6s to 10s]) with an overcard kicker'

except NameError:
    pass
dfPreflop


# In[257]:


HAND_STRENGTH


# In[258]:


if HAND_STRENGTH > 30:
    dfPreflop['REASONS TO CALL'] = 'Hand is too strong to fold'
dfPreflop


# In[259]:


CARDS


# In[260]:


HAND_STRENGTH


# In[261]:


#TOP TWO PAIR ON THE RIVER : GO ALL-IN


# In[262]:


################# MODULE 1.5: CALCULATING POT-ODDS #########################


# In[263]:


#### GOAL: SUM BET SIZES BEFORE YOUR MOVE


# In[264]:


########## MODULE TWO PART TWO : DECISIONS ON THE TURN ############


# In[265]:


COMMUNITY_CARDS 


# In[266]:


HAND = COMMUNITY_CARDS[0:2]
FLOP = COMMUNITY_CARDS[2:5]
BURN1 = COMMUNITY_CARDS[5]
TURN = COMMUNITY_CARDS[6]
BURN2 = COMMUNITY_CARDS[7]
RIVER = COMMUNITY_CARDS[8]
FLOP


# In[267]:


HAND


# In[268]:


TURN


# In[269]:


TURN_RANK


# In[270]:


FLOP_1_RANK


# In[271]:


FLOP_2_RANK


# In[272]:


FLOP_3_RANK


# In[273]:


dfPreflop['TURN CARD'] = str(TURN)

if TURN_RANK == FLOP_1_RANK or FLOP_2_RANK or FLOP_3_RANK:
    dfPreflop['REASONS TO CALL'] = 'You have at least a pair'
dfPreflop


# In[274]:


dfFlopTurn = dfPreflop
dfFlopTurn


# In[ ]:





# In[275]:


RAISE_POSSIBILITIES = ('Opponent plays very well/folds out hands you beat')


# In[276]:


########### MODULE TWO PART THREE : DECISIONS ON THE RIVER ############


# In[277]:


RIVER = COMMUNITY_CARDS[8]


# In[278]:


RIVER


# In[279]:


dfFlopTurn['RIVER'] = RIVER
dfFlopTurn


# In[280]:


RIVER


# In[281]:


#Write this as a list :'[x]'
NUMBER_OF_PLAYERS = [4]


# In[282]:


NUMBER_OF_PLAYERS


# In[283]:


FLOP


# In[284]:


EQUITY = ('backdoor flush draw','gutshot straight draw','overcards')


# In[285]:


EQUITY


# In[286]:


#Table Player Type and Stack Sizes 
print("Q600")
print('''
                               ''' +CPU5+       '''              ''' +CPU4+'''  
                               ''' +STACK5+'''   '''+STACK4+'''
                                       _____________________________
                                    /                                \                           
                      ''' +CPU8+'''                        ''' +CPU1+'''    
                      ''' +STACK8+'''              ''' +STACK1+'''  
                                  /                                    \ 
                                  |                                     |
                      ''' +CPU7+'''                        ''' +CPU2+'''
                      ''' +STACK7+'''                ''' +STACK2+'''
                                  |                                     | 
                                   \                                    /
                      ''' +CPU6+'''                        ''' +CPU3+'''
                      ''' +STACK6+'''                 ''' +STACK3+'''
                                     \                                /
                                       ------------------------------
                                                   HERO
   ''') 


# In[287]:


print('''
                               ''' +CPU5+''':'''+CPU5_SITUATION+'''   ''' +CPU4+''':'''+CPU5_SITUATION+'''                                              
                                       _____________________________
                                    /                                \                           
                      ''' +CPU8+''':'''+CPU5_SITUATION+''' '''+CPU1+''':'''+CPU5_SITUATION+'''           
                                  /                                    \ 
                                  |                                     |'''
                      + CPU7+''' :'''+CPU5_SITUATION+'''+'''                         +CPU2+''':'''+CPU5_SITUATION+'''
                                  |                                     | 
                                   \                                    /
                      ''' +CPU6+''':'''+CPU5_SITUATION+'''+'''                         +CPU3+''':'''+CPU5_SITUATION+'''
                                     \                                /
                                       ------------------------------
                                                   YOU
                 Position: '''+str(My_Position) +' ~ ' +str(My_Situation[1]) + " ~ YOUR MOVE"+'''
   ''') 


# In[288]:


#CAll PREFLOP SB 3.5 TO 1 POT ODDS BECAUSE THERE'S A $2 $2 $2 bet treain in 
# front of you and you only need $1 to call, hand can get top pair sometimes

# dfPreflop.loc(0)['REASONS TO CALL'] = dfPreflop['']
# dfPreflop.at["Preflop","REASONS TO CALL"] = FLOP

dfPreflop['REASONS TO CALL'] = 'str' + dfPreflop['REASONS TO CALL'].astype(str)
dfPreflop

dfPreflop['REASONS TO BET CALL'] = dfPreflop['POT ODDS'].apply(lambda x: 
    '''Call Preflop when you have 3.5 to 1 pot odds after 3 previous same-size bets and one caller behind, especially in the Small Blind''' 
    #########FIRST TO ACT INCLUDING FOLDS:
        if MY_STACK_SIZE == "Stack is >150 Big Blinds" 
            and HAND_STRENGTH >= 30
                and My_Position == 'UTG'
                    or MY_STACK_SIZE == "Stack is >150 Big Blinds" 
                        and x == 'UTG'
                            and HAND_STRENGTH >= 30
                                else np.NaN)


# In[289]:


# LOAD ALL CARD IMAGES

from PIL import Image, ImageDraw, ImageFilter

basewidth = 75

imgAh = Image.open("Ace of Hearts.png") 
wpercent = (basewidth/float(imgAh.size[0]))
hsize = int((float(imgAh.size[1])*float(wpercent)))
imgAh =imgAh.resize((basewidth,hsize), Image.LANCZOS)
imgAh

img2h = Image.open("Two of Hearts.png")
wpercent = (basewidth/float(img2h.size[0])) 
hsize = int((float(img2h.size[1])*float(wpercent))) 
img2h =img2h.resize((basewidth,hsize), Image.LANCZOS) 
img2h
img3h = Image.open("Three of Hearts.png") 
wpercent = (basewidth/float(img3h.size[0])) 
hsize = int((float(img3h.size[1])*float(wpercent))) 
img3h =img3h.resize((basewidth,hsize), Image.LANCZOS) 
img3h
img4h = Image.open("Four of Hearts.png")
wpercent = (basewidth/float(img4h.size[0])) 
hsize = int((float(img4h.size[1])*float(wpercent))) 
img4h =img4h.resize((basewidth,hsize), Image.LANCZOS) 
img4h
img5h = Image.open("Five of Hearts.png")
wpercent = (basewidth/float(img5h.size[0])) 
hsize = int((float(img5h.size[1])*float(wpercent))) 
img5h =img5h.resize((basewidth,hsize), Image.LANCZOS) 
img5h
img6h = Image.open("Six of Hearts.png") 
wpercent = (basewidth/float(img6h.size[0])) 
hsize = int((float(img6h.size[1])*float(wpercent))) 
img6h =img6h.resize((basewidth,hsize), Image.LANCZOS) 
img6h
img7h = Image.open("Seven of Hearts.png") 
wpercent = (basewidth/float(img7h.size[0])) 
hsize = int((float(img7h.size[1])*float(wpercent))) 
img7h =img7h.resize((basewidth,hsize), Image.LANCZOS) 
img7h
img8h = Image.open("Eight of Hearts.png")
wpercent = (basewidth/float(img8h.size[0])) 
hsize = int((float(img8h.size[1])*float(wpercent))) 
img8h =img8h.resize((basewidth,hsize), Image.LANCZOS) 
img8h
img9h = Image.open("Nine of Hearts.png") 
wpercent = (basewidth/float(img9h.size[0])) 
hsize = int((float(img9h.size[1])*float(wpercent))) 
img9h =img9h.resize((basewidth,hsize), Image.LANCZOS) 
img9h
img10h = Image.open("Ten of Hearts.png")
wpercent = (basewidth/float(img10h.size[0])) 
hsize = int((float(img10h.size[1])*float(wpercent))) 
img10h =img10h.resize((basewidth,hsize), Image.LANCZOS) 
img10h
imgJh = Image.open("Jack of Hearts.png")
wpercent = (basewidth/float(imgJh.size[0])) 
hsize = int((float(imgJh.size[1])*float(wpercent))) 
imgJh =imgJh.resize((basewidth,hsize), Image.LANCZOS) 
imgJh
imgQh = Image.open("Queen of Hearts.png") 
wpercent = (basewidth/float(imgQh.size[0])) 
hsize = int((float(imgQh.size[1])*float(wpercent))) 
imgQh =imgQh.resize((basewidth,hsize), Image.LANCZOS) 
imgQh
imgKh = Image.open("King of Hearts.png")
wpercent = (basewidth/float(imgKh.size[0])) 
hsize = int((float(imgKh.size[1])*float(wpercent))) 
imgKh =imgKh.resize((basewidth,hsize), Image.LANCZOS) 
imgKh
imgAc = Image.open("Ace of Clubs.png")
wpercent = (basewidth/float(imgAc.size[0])) 
hsize = int((float(imgAc.size[1])*float(wpercent))) 
imgAc =imgAc.resize((basewidth,hsize), Image.LANCZOS) 
imgAc
img2c = Image.open("Two of Clubs.png")
wpercent = (basewidth/float(img2c.size[0])) 
hsize = int((float(img2c.size[1])*float(wpercent))) 
img2c =img2c.resize((basewidth,hsize), Image.LANCZOS) 
img2c
img3c = Image.open("Three of Clubs.png")
wpercent = (basewidth/float(img3c.size[0])) 
hsize = int((float(img3c.size[1])*float(wpercent))) 
img3c =img3c.resize((basewidth,hsize), Image.LANCZOS) 
img3c
img4c = Image.open("Four of Clubs.png") 
wpercent = (basewidth/float(img4c.size[0])) 
hsize = int((float(img4c.size[1])*float(wpercent))) 
img4c =img4c.resize((basewidth,hsize), Image.LANCZOS) 
img4c
img5c = Image.open("Five of Clubs.png")
wpercent = (basewidth/float(img5c.size[0])) 
hsize = int((float(img5c.size[1])*float(wpercent))) 
img5c =img5c.resize((basewidth,hsize), Image.LANCZOS) 
img5c
img6c = Image.open("Six of Clubs.png")
wpercent = (basewidth/float(img6c.size[0])) 
hsize = int((float(img6c.size[1])*float(wpercent))) 
img6c =img6c.resize((basewidth,hsize), Image.LANCZOS) 
img6c
img7c = Image.open("Seven of Clubs.png")
wpercent = (basewidth/float(img7c.size[0])) 
hsize = int((float(img7c.size[1])*float(wpercent))) 
img7c =img7c.resize((basewidth,hsize), Image.LANCZOS) 
img7c
img8c = Image.open("Eight of Clubs.png")
wpercent = (basewidth/float(img8c.size[0])) 
hsize = int((float(img8c.size[1])*float(wpercent))) 
img8c =img8c.resize((basewidth,hsize), Image.LANCZOS) 
img8c
img9c = Image.open("Nine of Clubs.png") 
wpercent = (basewidth/float(img9c.size[0])) 
hsize = int((float(img9c.size[1])*float(wpercent))) 
img9c =img9c.resize((basewidth,hsize), Image.LANCZOS) 
img9c
img10c = Image.open("Ten of Clubs.png")
wpercent = (basewidth/float(img10c.size[0])) 
hsize = int((float(img10c.size[1])*float(wpercent))) 
img10c =img10c.resize((basewidth,hsize), Image.LANCZOS) 
img10c
imgJc = Image.open("Jack of Clubs.png")
wpercent = (basewidth/float(imgJc.size[0])) 
hsize = int((float(imgJc.size[1])*float(wpercent))) 
imgJc =imgJc.resize((basewidth,hsize), Image.LANCZOS) 
imgJc
imgQc = Image.open("Queen of Clubs.png") 
wpercent = (basewidth/float(imgQc.size[0])) 
hsize = int((float(imgQc.size[1])*float(wpercent))) 
imgQc =imgQc.resize((basewidth,hsize), Image.LANCZOS) 
imgQc
imgKc = Image.open("King of Clubs.png")
wpercent = (basewidth/float(imgKc.size[0])) 
hsize = int((float(imgKc.size[1])*float(wpercent))) 
imgKc =imgKc.resize((basewidth,hsize), Image.LANCZOS) 
imgKc
imgAs = Image.open("Ace of Spades.png")
wpercent = (basewidth/float(imgAs.size[0])) 
hsize = int((float(imgAs.size[1])*float(wpercent))) 
imgAs =imgAs.resize((basewidth,hsize), Image.LANCZOS) 
imgAs
img2s = Image.open("Two of Spades.png") 
wpercent = (basewidth/float(img2s.size[0])) 
hsize = int((float(img2s.size[1])*float(wpercent))) 
img2s =img2s.resize((basewidth,hsize), Image.LANCZOS) 
img2s
img3s = Image.open("Three of Spades.png")
wpercent = (basewidth/float(img3s.size[0])) 
hsize = int((float(img3s.size[1])*float(wpercent))) 
img3s =img3s.resize((basewidth,hsize), Image.LANCZOS) 
img3s
img4s = Image.open("Four of Spades.png") 
wpercent = (basewidth/float(img4s.size[0])) 
hsize = int((float(img4s.size[1])*float(wpercent))) 
img4s =img4s.resize((basewidth,hsize), Image.LANCZOS) 
img4s
img5s = Image.open("Five of Spades.png") 
wpercent = (basewidth/float(img5s.size[0])) 
hsize = int((float(img5s.size[1])*float(wpercent))) 
img5s =img5s.resize((basewidth,hsize), Image.LANCZOS) 
img5s
img6s = Image.open("Six of Spades.png") 
wpercent = (basewidth/float(img6s.size[0])) 
hsize = int((float(img6s.size[1])*float(wpercent))) 
img6s =img6s.resize((basewidth,hsize), Image.LANCZOS) 
img6s
img7s = Image.open("Seven of Spades.png") 
wpercent = (basewidth/float(img7s.size[0])) 
hsize = int((float(img7s.size[1])*float(wpercent))) 
img7s =img7s.resize((basewidth,hsize), Image.LANCZOS) 
img7s
img8s = Image.open("Eight of Spades.png") 
wpercent = (basewidth/float(img8s.size[0])) 
hsize = int((float(img8s.size[1])*float(wpercent))) 
img8s =img8s.resize((basewidth,hsize), Image.LANCZOS) 
img8s
img9s = Image.open("Nine of Spades.png") 
wpercent = (basewidth/float(img9s.size[0])) 
hsize = int((float(img9s.size[1])*float(wpercent))) 
img9s =img9s.resize((basewidth,hsize), Image.LANCZOS) 
img9s
img10s = Image.open("Ten of Spades.png") 
wpercent = (basewidth/float(img10s.size[0])) 
hsize = int((float(img10s.size[1])*float(wpercent))) 
img10s =img10s.resize((basewidth,hsize), Image.LANCZOS) 
img10s
imgJs = Image.open("Jack of Spades.png") 
wpercent = (basewidth/float(imgJs.size[0])) 
hsize = int((float(imgJs.size[1])*float(wpercent))) 
imgJs =imgJs.resize((basewidth,hsize), Image.LANCZOS) 
imgJs
imgQs = Image.open("Queen of Spades.png")
wpercent = (basewidth/float(imgQs.size[0])) 
hsize = int((float(imgQs.size[1])*float(wpercent))) 
imgQs =imgQs.resize((basewidth,hsize), Image.LANCZOS) 
imgQs
imgKs = Image.open("King of Spades.png")
wpercent = (basewidth/float(imgKs.size[0])) 
hsize = int((float(imgKs.size[1])*float(wpercent))) 
imgKs =imgKs.resize((basewidth,hsize), Image.LANCZOS) 
imgKs
imgAd = Image.open("Ace of Diamonds.png") 
wpercent = (basewidth/float(imgAd.size[0])) 
hsize = int((float(imgAd.size[1])*float(wpercent))) 
imgAd =imgAd.resize((basewidth,hsize), Image.LANCZOS) 
imgAd
img2d = Image.open("Two of Diamonds.png")
wpercent = (basewidth/float(img2d.size[0])) 
hsize = int((float(img2d.size[1])*float(wpercent))) 
img2d =img2d.resize((basewidth,hsize), Image.LANCZOS) 
img2d
img3d = Image.open("Three of Diamonds.png")
wpercent = (basewidth/float(img3d.size[0])) 
hsize = int((float(img3d.size[1])*float(wpercent))) 
img3d =img3d.resize((basewidth,hsize), Image.LANCZOS) 
img3d
img4d = Image.open("Four of Diamonds.png") 
wpercent = (basewidth/float(img4d.size[0])) 
hsize = int((float(img4d.size[1])*float(wpercent))) 
img4d =img4d.resize((basewidth,hsize), Image.LANCZOS) 
img4d
img5d = Image.open("Five of Diamonds.png")
wpercent = (basewidth/float(img5d.size[0])) 
hsize = int((float(img5d.size[1])*float(wpercent))) 
img5d =img5d.resize((basewidth,hsize), Image.LANCZOS) 
img5d
img6d = Image.open("Six of Diamonds.png")
wpercent = (basewidth/float(img6d.size[0])) 
hsize = int((float(img6d.size[1])*float(wpercent))) 
img6d =img6d.resize((basewidth,hsize), Image.LANCZOS) 
img6d
img7d = Image.open("Seven of Diamonds.png") 
wpercent = (basewidth/float(img7d.size[0])) 
hsize = int((float(img7d.size[1])*float(wpercent))) 
img7d =img7d.resize((basewidth,hsize), Image.LANCZOS) 
img7d
img8d = Image.open("Eight of Diamonds.png") 
wpercent = (basewidth/float(img8d.size[0])) 
hsize = int((float(img8d.size[1])*float(wpercent))) 
img8d =img8d.resize((basewidth,hsize), Image.LANCZOS) 
img8d
img9d = Image.open("Nine of Diamonds.png")
wpercent = (basewidth/float(img9d.size[0])) 
hsize = int((float(img9d.size[1])*float(wpercent))) 
img9d =img9d.resize((basewidth,hsize), Image.LANCZOS) 
img9d
img10d = Image.open("Ten of Diamonds.png")
wpercent = (basewidth/float(img10d.size[0])) 
hsize = int((float(img10d.size[1])*float(wpercent))) 
img10d =img10d.resize((basewidth,hsize), Image.LANCZOS) 
img10d
imgJd = Image.open("Jack of Diamonds.png")
wpercent = (basewidth/float(imgJd.size[0])) 
hsize = int((float(imgJd.size[1])*float(wpercent))) 
imgJd =imgJd.resize((basewidth,hsize), Image.LANCZOS) 
imgJd
imgQd = Image.open("Queen of Diamonds.png")
wpercent = (basewidth/float(imgQd.size[0])) 
hsize = int((float(imgQd.size[1])*float(wpercent))) 
imgQd =imgQd.resize((basewidth,hsize), Image.LANCZOS) 
imgQd
imgKd = Image.open("King of Diamonds.png")
wpercent = (basewidth/float(imgKd.size[0])) 
hsize = int((float(imgKd.size[1])*float(wpercent))) 
imgKd =imgKd.resize((basewidth,hsize), Image.LANCZOS) 
basewidth = 75
imgJoker = Image.open("trump joker image.png")
wpercent = (basewidth/float(imgJoker.size[0]))
hsize = int((float(imgJoker.size[1])*float(wpercent)))
imgJoker = imgJoker.resize((basewidth,hsize), Image.LANCZOS)
imgRectangle = Image.open("BLACK RECTANGLE.png")
wpercent = (basewidth/float(imgRectangle.size[0]))
hsize = int((float(imgRectangle.size[1])*float(wpercent)))
imgRectangle = imgRectangle.resize((basewidth,hsize), Image.LANCZOS)
imgBLUESQUARE = Image.open("LIGHT BLUE SQUARE.png")
wpercent = (basewidth/float(imgBLUESQUARE.size[0]))
hsize = int((float(imgBLUESQUARE.size[1])*float(wpercent)))
imgBLUESQUARE = imgBLUESQUARE.resize((basewidth,hsize), Image.LANCZOS)
# LOAD ALL CARD IMAGES (step ONE)

print("CARD IMAGES LOADED!")


# In[290]:


# ALL CARDS with img first
CARD_IMAGES = (imgAh, img2h, img3h, img4h, img5h, img6h, img7h, img8h, img9h, img10h, imgJh, imgQh, imgKh, imgAc, img2c, img3c, img4c, img5c, img6c, img7c, img8c, img9c, img10c, imgJc, imgQc, imgKc, imgAs, img2s, img3s, img4s, img5s, img6s, img7s, img8s, img9s, img10s, imgJs, imgQs, imgKs, imgAd, img2d, img3d, img4d, img5d, img6d, img7d, img8d, img9d, img10d, imgJd, imgQd, imgKd)
print("CARD IMAGES ASSIGNED TO VARIABLES!")


# In[291]:


# HEADS UP CARD DISTRIBUTION

import random

RANDOMIZED_CARDS = random.sample((CARD_IMAGES), 52)

HOLE_1_PLAYER_1 = RANDOMIZED_CARDS[0]
HOLE_1_PLAYER_2 = RANDOMIZED_CARDS[1] 
HOLE_2_PLAYER_1 = RANDOMIZED_CARDS[2]
HOLE_2_PLAYER_2 = RANDOMIZED_CARDS[3]
BURN_1 = RANDOMIZED_CARDS[4]
FLOP_1 = RANDOMIZED_CARDS[5] 
FLOP_2 = RANDOMIZED_CARDS[6]
FLOP_3 = RANDOMIZED_CARDS[7]
BURN_2 = RANDOMIZED_CARDS[8]
TURN = RANDOMIZED_CARDS[9]
BURN_3 = RANDOMIZED_CARDS[10]
RIVER = RANDOMIZED_CARDS[11]

print("CARDS RANDOMIZED AND DEALT: TWO PLAYERS, BURN, FLOP, BURN, TURN, BURN, RIVER")


# In[292]:


# GREEN POKER FELT

im1 = Image.open('Green Felt.jpg')
im1


# In[293]:


im1.paste(imgRectangle,(490,75))
im1.paste(imgRectangle,(565,75))

im1.paste(imgRectangle,(490,375))
im1.paste(imgRectangle,(565,375))

#im1.paste(imCOLOR, (distance from left, distance from top))

im1.paste(imgBLUESQUARE,(640,75))
im1.paste(imgBLUESQUARE,(700,75))

im1.paste(imgBLUESQUARE,(700,375))
im1.paste(imgBLUESQUARE,(640,375))

im1


# In[294]:


CPU1_STACK_SIZE_PREFLOP = 100

HERO_STACK_SIZE_PREFLOP = 250

CPU1_STACK_SIZE_PREFLOP


# In[295]:


from PIL import Image, ImageDraw, ImageFont

# create an image

# get a font
fnt = ImageFont.truetype("Arial.ttf", 25)
# get a drawing context
d = ImageDraw.Draw(im1)

# draw multiline text

#CPU1 Attribute
d.multiline_text((500,25), "Loose, Aggressive Guy", font=fnt, fill=(250, 250, 250))
#CPU1 STACK SIZE
d.multiline_text((500,400), str(CPU1_STACK_SIZE_PREFLOP), font=fnt, fill=(250, 250, 250))
#CPU2 STACK SIZE
d.multiline_text((500,100), str(HERO_STACK_SIZE_PREFLOP), font=fnt, fill=(250, 250, 250))

im1



# In[296]:


#Opponent's Card #1
im1.paste(imgJoker,(300,75))

im1


# In[297]:


# Hero's Card #1
im1.paste(imgJoker,(300,350))
im1


# In[298]:


#Opponent's card 2
im1.paste(imgJoker,(400,75))
im1


# In[299]:


# POT 

im1.paste(imgRectangle,(640,235))
im1.paste(imgRectangle,(710,235))

d.multiline_text((650,250), str('POT'), font=fnt, fill=(250, 250, 250))


im1


# In[300]:


# Hero's Card 2
im1.paste(imgJoker,(400,350))


im1


# In[301]:


# Hole Card 1
im1.paste(HOLE_1_PLAYER_1,(300,350))
im1


# In[302]:


# Hole card 2 HERO
im1.paste(HOLE_2_PLAYER_1,(400,350))
im1


# In[303]:


# BETS PREFLOP

#BET 100 each

CPU1_BET_1 = 100

HERO_BET_1 = 100

CPU1_STACK_SIZE_FLOP = CPU1_STACK_SIZE_PREFLOP - CPU1_BET_1

HERO_STACK_SIZE_FLOP = HERO_STACK_SIZE_PREFLOP - HERO_BET_1

POT = -(CPU1_STACK_SIZE_FLOP - CPU1_STACK_SIZE_PREFLOP + (HERO_STACK_SIZE_FLOP - HERO_STACK_SIZE_PREFLOP))

POT


# In[304]:


#RE-PASTE BLACK SQUARES

im1.paste(imgRectangle,(500,75))
im1.paste(imgRectangle,(575,75))

im1.paste(imgRectangle,(500,375))
im1.paste(imgRectangle,(575,375))

#BET 1 - CPU AND HERO

d.multiline_text((600,400), str(CPU1_BET_1), font=fnt, fill=(250, 250, 250))
d.multiline_text((600,100), str(HERO_BET_1), font=fnt, fill=(250, 250, 250))

# POT 

im1.paste(imgRectangle,(640,235))
im1.paste(imgRectangle,(710,235))

d.multiline_text((650,250), str(POT), font=fnt, fill=(250, 250, 250))

#RE-PASTE STACK SIZES

# get a font
fnt = ImageFont.truetype("Arial.ttf", 40)
# get a drawing context
d = ImageDraw.Draw(im1)

# draw multiline text

d.multiline_text((650,400), str(CPU1_STACK_SIZE_FLOP), font=fnt, fill=(250, 250, 250))
d.multiline_text((650,100), str(HERO_STACK_SIZE_FLOP), font=fnt, fill=(250, 250, 250))

im1


# In[305]:


# Community Cards Back (Joker)
im1.paste(imgJoker,(250,220))
im1.paste(imgJoker,(150,220))
im1.paste(imgJoker,(350,220))
im1.paste(imgJoker,(450,220))
im1.paste(imgJoker,(550,220))
im1


# In[306]:


# Community Cards Front 
im1.paste(FLOP_1,(250,215))
im1.paste(FLOP_2,(150,215))
im1.paste(FLOP_3,(350,215))
im1


# In[307]:


im1.paste(TURN,(450,215))
im1


# In[308]:


im1.paste(RIVER,(550,215))

im1


# In[309]:


#Opponent's Cards
im1.paste(HOLE_1_PLAYER_2,(300,75))
im1.paste(HOLE_2_PLAYER_2,(400,75))
im1


# In[310]:


CPU1 = random.choice(['Loose, Aggressive Guy','Tight, Aggressive Guy','Loose, Passive Guy','Tight, Passive Guy'])
CPU1


# In[311]:


STACK1 = random.choice(['Short Stack (<=40 Big Blinds)','Medium Stack (41-90 Big Blinds)','Deep Stack (>=200 Big Blinds)','Big Stack (>= 91 Big Blinds)'])
STACK1


# In[312]:


HOLE_1_PLAYER_2


# In[313]:


HOLE_2_PLAYER_1


# In[314]:


HOLE_2_PLAYER_2


# In[315]:


BURN_1


# In[316]:


FLOP_1


# In[317]:


FLOP_2


# In[318]:


FLOP_3


# In[319]:


BURN_2


# In[320]:


TURN


# In[321]:


BURN_3


# In[322]:


RIVER


# In[323]:


#30 SECOND TIMER

import tkinter as tk

counter = 31 
def counter_label(label):
  counter = 0
  def count():
    global counter
    counter -= 1
    label.config(text=str(counter))
    label.after(1000, count)
  count()
 
 
root = tk.Tk()
root.title("30 Second Timer")
label = tk.Label(root, fg="dark green")
label.pack()
counter_label(label)
button = tk.Button(root, text='ACT', width=25, command=root.destroy)
button.pack()
root.mainloop()


# In[324]:


# Decision Choices from Fold to All-In

import tkinter as tk

def FOLD():
    print("YOU FOLD")
    
def CHECK():
    print("YOU CHECK")

def HALF_POT_SIZED_BET():
    print("YOU RAISE TO A 1/2 POT BET")

def POT_SIZED_BET():
    print("YOU RAISE TO A POT-SIZED BET")
    
def ALL_IN():
    print("YOU GO ALL-IN!")
    
root = tk.Tk()
frame = tk.Frame(root)
frame.pack()


button = tk.Button(frame, 
                   text="FOLD", 
                   fg="red",
                   command=FOLD)
button.pack(side=tk.LEFT)

button = tk.Button(frame, 
                   text="CHECK", 
                   fg="blue",
                   command=CHECK)
button.pack(side=tk.LEFT)

button = tk.Button(frame, 
                   text="HALF-POT SIZED BET", 
                   fg="black",
                   command=HALF_POT_SIZED_BET)
button.pack(side=tk.LEFT)

slogan = tk.Button(frame,
                   text="POT-SIZE BET",
                   fg='purple',
                   command=POT_SIZED_BET)
slogan.pack(side=tk.LEFT)

slogan = tk.Button(frame,
                   text="ALL-IN",
                   fg='GREEN',
                   command=ALL_IN)
slogan.pack(side=tk.LEFT)

root.mainloop()


# In[325]:


im1


# In[326]:


# import tkinter
# import tkMessageBox

# top = Tk()

# Lb1 = Listbox(top)
# Lb1.insert(1, "Python")
# Lb1.insert(2, "Perl")
# Lb1.insert(3, "C")
# Lb1.insert(4, "PHP")
# Lb1.insert(5, "JSP")
# Lb1.insert(6, "Ruby")

# Lb1.pack()
# top.mainloop()


# In[327]:


# STACK_SIZE_HERO
# STACK_SIZE_VILLAIN
# POT
# BET_HERO
# BET_VILLAIN

