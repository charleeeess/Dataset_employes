# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:49:44 2022

@author: 10161919
"""

import pandas as pd
import numpy as np


##IMPORTER unfichier Excel
CheminFichier = r"C:\Users\10161919\Desktop\Python\Dataset_Employes.xlsx"
Dataset_Employes = pd.read_excel(CheminFichier)


Dataset_Employes

#IMPORTER UN CSV
CheminFichierCSV = r"C:\Users\10161919\Desktop\Python\Dataset_Employes.csv"
Dataset_EmployesCSV = pd.read_csv(CheminFichierCSV, sep=';')

Dataset_EmployesCSV

#Types des colonnes 
#Retourne la forme du tableau de données (=Le dataset)
#Affichage du nombre de ligne et du nombre de colonne 
Dataset_EmployesCSV.shape

#Retourne le nombre de valeurs non manquantes ainsique les info de types de données
Dataset_Employes.info()

#Filtrer les données
#Filtre des contrats en CDI
Dataset_Employes['Contrat'].value_counts()

Data_Filtrée_CDI = Dataset_Employes['Contrat'] == 'CDI'
Data_Filtrée_CDI

Dataset_Employes_CDI = Dataset_Employes[Data_Filtrée_CDI]
Dataset_Employes_CDI 

#Filtre sur les durées de travail > 25 heures 

Dataset_Employes['Durée hebdo'].value_counts()

Data_Filtrée_durée = Dataset_Employes['Durée hebdo'] > 25
Data_Filtrée_durée


Data_Filtrée_durée_Plus25H = Dataset_Employes[Data_Filtrée_durée]
Data_Filtrée_durée_Plus25H 

#Importation des librairies de visualisations
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")

#Affichage de l'aide: documentation sur la fonction lineplot()
help(sns.lineplot)

x = np.linspace(0,2 * np.pi, 200)
y = np.sin(x)
sns.lineplot(x,y)

#Les courbes 
#Relation continue simple 
sns.lineplot(x = 'Age', y = 'Niveau de satisfaction', data = Dataset_Employes)
#Relation continue multiple
sns.relplot(x = 'Age', y = 'Niveau de satisfaction', hue = 'Type Contrat', data = Dataset_Employes, kind = 'line', ci = None)
#Nous remarquons que nous n'avons pas de lien évident entre l'âge des salariés et leur niveau de satisfaction et ce peu importe le type de contrat

Dataset_Employes['Service'].value_counts()

#Diagramme en Barres simples 

sns.set(rc={'figure.figsize':(10,3)})
sns.countplot(x= 'Service', data = Dataset_Employes)

#Diagramme en barres regroupées
sns.catplot(x= 'Service', y = 'Salaire base mensuel', hue = 'Contrat', data = Dataset_Employes, kind='bar', ci=None, palette = sns.color_palette('Set2'))

#Histogramme
#En statistique, un histogramme est une représentation graphique permettant de visualiser la répartition empirique d'une variable ou chacune des barres représente un interval de valeurs
sns.distplot(Dataset_Employes['Niveau de satisfaction']
sns.distplot(Dataset_Employes['Niveau de satisfaction'], bins = 20) #bins pour augmenter ou diminuer le nombre de barre
sns.distplot(Dataset_Employes['Niveau de satisfaction'], kde= False , bins = 20) #kde pour enlever la courbe 

#Diagramme à moustache (Boxplot) permet de résumer quelques indicateurs statistique tel que les quartiles, la médiane, le maximum
Dataset_Employes.describe()

sns.boxplot(Dataset_Employes['Age'])

sns.boxplot(x = Dataset_Employes['Age'], y = Dataset_Employes['Service'])

sns.set(style='whitegrid', palette='pastel')
sns.boxplot(x= 'Age', y = 'Service', hue= 'Type Contrat', data = Dataset_Employes, palette=["m","g"])


#Nuage de points basique
sns.relplot(x= 'Age', y = 'Niveau de satisfaction', data = Dataset_Employes)

#Visualisons la droite de régression 
sns.lmplot(x= 'Age', y = 'Ancienneté_an', data = Dataset_Employes) #Légère courbe ce qui signifie qu'il y'a tout de même un léger lien

#Ajouter des dimensions au nuage de points
#TROIS DIMENSIONS
sns.relplot(x= 'Age', y = 'Niveau de satisfaction', hue = 'Type Contrat', data = Dataset_Employes)

#QUATRE DIMENSIONS
sns.relplot(x= 'Age', y = 'Niveau de satisfaction', hue = 'Type Contrat', size = 'Salaire base mensuel', data = Dataset_Employes)

#CINQ DIMENSIONS
sns.relplot(x= 'Age', y = 'Niveau de satisfaction', hue = 'Type Contrat', size = 'Salaire base mensuel', style = 'Contrat', data = Dataset_Employes)


#INTRODUCTION 
#IMPORT DES LIBRAIRIES
#Indique à matplotlib d'afficher les graphiques 'en ligne'
%matplotlib inline
#Importation des librairies de visualisations
from matplotlib import pyplot as plt

#Aide grâce à la fontion:
help(plt.plot)

#Premier graphique 
x = np.linspace(0, 10 * np.pi, 200)
y = np.sin(x)

fig, ax = plt.subplots() #Permet la création de la figure et des axes pour la fenêtre graphique 
ax.plot(x, y)
plt.show()

#Utiliser des paramètres pour changer l'aspect des courbes

#Changer la couleur 

plt.plot(x,y,color='green')
plt.show()

#Changer le type de ligne 

plt.plot(x,y,color='blue', linestyle='dotted')
plt.show()

plt.plot(x,y,color='blue', linestyle=(0,(3,10,1,10,1,10)))
plt.show()

#Superposer des courbes 
#Création des données

import numpy as np

x= np.arange(0.0, 2.0, 0.01)
y1 = 1 + np.sin(2 * np.pi * x)
y2 = np.cos(2 * np.pi * x)

plt.plot(x,y1,color='red')
plt.plot(x,y2,color='green')
plt.show()

Data_Group = Dataset_Employes.groupby(['Niveau de satisfaction'])['Age','Ancienneté_an'].mean()
Data_Group = Data_Group.reset_index()
Data_Group

plt.plot(Data_Group['Niveau de satisfaction'], Data_Group['Age'], color='red')
plt.plot(Data_Group['Niveau de satisfaction'], Data_Group['Ancienneté_an'], color='green')
plt.show()

#Les diagrammes en barres empilées et horizontaux

#Diagramme en barre classique 

DF = Dataset_Employes['Contrat'].value_counts()
DF

DF.values
DF.index

plt.bar(DF.index,DF.values)
plt.show()

#Diagramme en barre horizontales
plt.barh(DF.index, DF.values)
plt.show()

#Diagramme en barres empilées

DF = Data_Group = Dataset_Employes.groupby(['Contrat'])['Promotion','Augmentation'].sum()
DF = pd.DataFrame(DF).reset_index()
DF

plt.bar(DF['Contrat'], DF['Augmentation'])
plt.bar(DF['Contrat'], DF['Promotion'], bottom=DF['Augmentation'])
plt.show()

#Les diagrammes en secteurs (PieChart)
DFPie = Dataset_Employes['Service'].value_counts(normalize=True).mul(100)
DFPie

Nomservice = DFPie.keys().tolist()
Nomservice

valeurs = DFPie.tolist()
valeurs

plt.pie(valeurs, labels = Nomservice, autopct='%1.1F%%')
plt.show()

#explode
explode = (0, 0.1, 0, 0 ,0 ,0)
plt.pie(valeurs, labels = Nomservice, autopct='%1.2F%%', explode=explode, shadow= True, startangle=90)
plt.show()

#Ajouter une légende et un titre

DF = Data_Group = Dataset_Employes.groupby(['Contrat'])['Promotion','Augmentation'].sum()
DF = pd.DataFrame(DF).reset_index()
DF

plt.bar(DF['Contrat'], DF['Augmentation'])
plt.bar(DF['Contrat'], DF['Promotion'], bottom=DF['Augmentation'])

plt.legend(['Augmentation','Promotion'], title = 'Légende', title_fontsize = 12, loc='upper left')

plt.title("Répartition des augmentations et promotions par type de contrat")

plt.xlabel('Contrat')
plt.ylabel('Nombre')

plt.show()

#Ajouter un titre


plt.bar(DF['Contrat'], DF['Augmentation'])
plt.bar(DF['Contrat'], DF['Promotion'], bottom=DF['Augmentation'])
plt.title("Répartition des augmentations et promotions par type de contrat")
plt.legend(['Augmentation','Promotion'], title = 'Légende', title_fontsize = 12, loc='upper left')
plt.xlabel('Contrat')
plt.ylabel('Nombre')

plt.text(DF['Contrat'][0], 0, DF['Augmentation'][0]) # Ajout de text sur Les barres
plt.text(DF['Contrat'][0], 10, DF['Promotion'][0])
plt.text(DF['Contrat'][1], 122, DF['Augmentation'][1])
plt.text(DF['Contrat'][1], 240, DF['Promotion'][1])

plt.show()


#Agrandisement de la fenêtre 
plt.figure(figsize = (10,10))

DFPie = Dataset_Employes['Service'].value_counts(normalize=True).mul(100)
Nomservice = DFPie.keys().tolist()
valeurs = DFPie.tolist()

DFPie
DFPie.keys()
Nomservice
valeurs

plt.pie(valeurs,labels=Nomservice, autopct='%1.1f%%')
plt.title("Proportion des employés par service", fontweight="bold")

plt.show()

#Faire coincider des graphique sur une même fenêtre 

plt.figure(figsize=(10,10))

#Création des courbes
plt.subplot(221)
plt.plot(Data_Group['Niveau de satisfaction'],Data_Group['Age'], color = 'red')
plt.plot(Data_Group['Niveau de satisfaction'],Data_Group['Ancienneté_an'], color = 'green')

#Création du piechart
plt.subplot(222)
plt.pie(valeurs,labels=Nomservice, autopct='%1.1f%%')
plt.title("Proportion des employés par service", fontweight="bold")

#Création du diagramme en barres
DF = Data_Group = Dataset_Employes.groupby(['Contrat'])['Promotion','Augmentation'].sum()
DF = pd.DataFrame(DF).reset_index()

plt.subplot(223)
plt.bar(DF['Contrat'], DF['Augmentation'])
plt.bar(DF['Contrat'], DF['Promotion'], bottom=DF['Augmentation'])
plt.title("Répartition des augmentations et promotions par type de contrat")

plt.show()


