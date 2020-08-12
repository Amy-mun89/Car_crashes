# Introduction
The City of Chicago believes that even one life lost in a traffic crash is unacceptable. We all have the right to walk, bike, take public transit, and drive on streets that are safe for everyone, regardless of who we are or where we live. That’s why we commit ourselves to-gether with Vision Zero to prevent death and serious injury from traffic crashes throughout our work and mutual effort. 
Traffic crashes are not “accidents.” We acknowledge that traffic deaths are preventable and unacceptable and commit to using all available tools to influence the conditions and behaviours that lead to serious crashes. Working together, we can eliminate traffic deaths and life-altering injuries.

# Data understanding
We work with the data which is supported by Chicago Data portal. (https://data.cityofchicago.org)  We used three datasets for this project, Traffic Crashes – crashes, Traffic Crashes – people, Traffic Crashes – Vehicle. The dataset is updated weekly base, and we used crashes only from October first of 2017 to October first of 2019 and it was adequate for building up models and analysis and data visualization. 
Each dataset has 48(crashes), 71(Vehicle), 29(people) column and ~360k(crashes), ~700k(vehicle), and ~770k(people), so we started the project by creating sample data from each of dataset and understanding the meaning of each columns so that we can reduce the data size and select the most important data. For the data understanding and sample dataset created, we used Python, Microsoft Excel, and tableau desktop.

## 1)	Data import and creation of sample dataset 
Since our raw data is big, we decide to make a sample data first and to work on the sample data for productivity. If the analysis or model quality is bad which is possible be-cause of the dataset size, we will create a new bigger sample data. However, working with a smaller sample dataset did not bring additional problems and made it simpler for the model and analysis.
Firstly, we downloaded the three datasets in the csv format. (crashes.csv, vehicles.csv, people.csv) and filtered the data only from 2017 October. (we only going to use the data from 2017 October) and create the sample dataset from crash data. The sample data is 10% from the original data and it was randomly picked by numpy random.rand function. Sec-ondly, we associated vehicles and people data by RD_NO. 
Now we have three sample datasets, crashes.sample.csv, vehicles.sample.csv, peo-ple.sample.csv and would working on with the sample dataset. 

## 2) Data understanding 
It is important to understand and check the meaning, description and type of each col-umns from each dataset from the table below to have a deeper and better understanding of the data. Some columns which have categorical value will be analysed in detail after the basic data understanding, and almost half of the features were removed after analysis and handling missing values


