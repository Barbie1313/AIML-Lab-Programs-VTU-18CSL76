import pandas as pd
from pandas import DataFrame

df_tennis=pd.read_csv("C:\\Users\\Adithi\\Desktop\\playtenniss.csv"\\playtenniss.csv")

print(df_tennis)

def entropy(probs):

    import math

    return sum([-prob*math.log(prob, 2) for prob in probs])

def entropy_of_list(a_list):

    from collections import Counter

    cnt=Counter(x for x in a_list)

    print("NO AND YES CLASSES: ",a_list.name,cnt)

    num_instances=len(a_list)*1.0

    probs=[x/num_instances for x in cnt.values()]

    return entropy(probs)

total_entropy=entropy_of_list(df_tennis['Target'])

print("ENTROPY OF GIVEN PLAYTENNIS DATA SET: ",total_entropy)

def information_gain(df,split_attribute_name,target_attribute_name,trace=0): 
    print("INFORMATION GAIN CALCULATION OF ",split_attribute_name)
    df_split=df.groupby(split_attribute_name)

    for name, group in df_split:

        print(name)

        print(group)

    nobs=len (df.index)*1.0

    df_agg_ent=df_split.agg({target_attribute_name: [entropy_of_list, lambda x: len(x)/nobs]})[target_attribute_name]

    df_agg_ent.columns=['Entropy', 'PropObservations']
    new_entropy=sum(df_agg_ent['Entropy']*df_agg_ent['PropObservations'])

    old_entropy=entropy_of_list(df[target_attribute_name])

    return (old_entropy-new_entropy)

print("INFO-GAIN FOR OUTLOOK IS: "+str(information_gain (df_tennis, "Outlook", "Target")), "\n")

print("INFO-GAIN FOR HUMIDITY IS: "+str(information_gain(df_tennis, "Humidity", "Target")), "\n")
print("INFO-GAIN FOR WIND IS: "+str(information_gain(df_tennis, "Wind", "Target")), "\n")

print("INFO-GAIN FOR TEMPERATURE IS: "+str(information_gain(df_tennis, "Temperature", "Target")), "\n")