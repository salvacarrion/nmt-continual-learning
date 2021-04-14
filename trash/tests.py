from torchtext.data.metrics import bleu_score
import sacrebleu

hyp = [
    "In patients with patients with patients with patients with chronic dermatoses is the disease is recommended the immune response and immunomodulation can be useful as part of treatment .",
    "Twenty @-@ eight patients and eight patients ( 8 patients ( 41.8 patients ( 8 patients ( 41.8 % developed hypotension ( SBP ( 20 % higher than 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value of 20 % basal value 20 % basal value of 20 % basal value of 20 % basal value 20 % basal value of 20 % basal value of 20 % basal value of 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value of 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value 20 % basal value",
    "CASE REPORT : LASE @-@ butter study of 31 years old with the aortic arch change and metallic prosthetic prosthetic prosthetic prosthetic prostheses in aortic valve in aortic valve in aortic valve .",
    "The percentage of diabetic tibial test was 1 that met the target group of the target group of GDSM @-@ IV gained metal goal was 90,0 % of @-@ 6 years old , and 66.66 % over 12 years old , the general fulfillment of the general fulfillment of the general fulfillment of the general degree of adherence of fulfillment of the general fulfillment of the general of the general of the general of the general fulfillment of the general of the general degree of the 76,71 % of the most of the most of the most of the most of the most of the most of the most of the most of the most of the most of the most of the most important degree of the most of the most of the most of the most of the most of the most of the most of the most of the most of the most of the studied group of the studied group of the most of the most important ones studied group of the most of the studied group of",
    "The oral cells can be considered as biological and biological qualities such as well as any brings , as cellultimate , are difficult and find to find any other site .",
    "Parcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcoparcop@@",
    "On the percentage of vices of vices was higher as is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is made is"]

hyp_g = [
    "In patients with patients with patients with skin disease and their disease , they are advised to stop the immune response , and message patients with a certain immune deficiency ." ,
    "Systolic blood pressure ( SBP ) decreased by 20 % from baseline .",
    "Case: The case is a 31-year-old pregnant @ - @ year-old who underwent aortic arch replacement , metallic valve for TA four years ago .",
    "The success rate for diabetic hemoglobin goals is 90.00 % from 0 @ - @ 6 years, 90.47 % from 6 @ - @ 12 general, and 66.66 % over 12 years. The overall success rate is 85.71 % ." ,
    "Like any famous person , they are rare and hard to find in the latest news .",
    "The first term plays a major role in the vulnerability model , although its source is unknown .",
    "Moreover , the percentage of wins also increases with the increase in chemical weapons .",
]

ref = [
    "In patients with chronic and relapsing dermatophytosis , the immune response evaluation is recommended , and immunomodulation could be useful as a rational measure in patients with a particular immunodeficiency .",
    "Twenty @-@ eight ( 41.8 % ) patients developed hypotension with their systolic blood pressure ( SBP ) decreasing &gt; 20 % of baseline .",
    "CASE REPORT : This is a 31 @-@ year old gravida who underwent exchange of the aortic arch and placement of a metallic aortic valve for TA four years ago .",
    "The percentage of success in glycated hemoglobin goals are 90,00 % from 0 @-@ 6 years , 90.47 % from 6 @-@ 12 years , and 66.66 % over 12 years . The general success percentage is 85.71 % .",
    "Like any celebrity , they are rare and hard to find anywhere else other than news headlines .",
    "The former term plays a key role in the frailty model , although its source is unknown .",
    "Moreover , the percentage of victories also gets higher as the CW increases ."
]

asdasd = 3





