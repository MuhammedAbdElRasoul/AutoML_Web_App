# conda activate pycaret_env
# streamlit run c:/Users/DELL/Desktop/apply/AutoML.py

import pandas as pd,  numpy as np, matplotlib.pyplot as plt, seaborn as sns, plotly.express as px

# Building the streamlit web app 
import streamlit as st
with st.sidebar :
    st.image("data:image/png;base64,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")
    st.title("AutoML Web Application")
    st.info("The purpose of this site is to provide general information about the hot new field of automated machine learning (AutoML) and software for AutoML using Python and the scikit-learn machine learning library.")
    choice = st.radio("Choose an operation: ",["Uploading & Reading a dataset","Performing an EDA","Data Preparing & Modelling"])

# Out the sidebar do the following :

# Check if the file exists or not ?.
import os
if  os.path.exists("dataset.csv") :
    df = pd.read_csv("dataset.csv", index_col=None)

# Make Choices    
if choice == "Uploading & Reading a dataset" :
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")

    # Check the file and upload it to the streamlit  
    if file :
        df = pd.read_csv(file, index_col = None)
        df.to_csv("dataset.csv", index=None)
        st.dataframe(df)

# Making an EDA 
if choice == "Performing an EDA" :
    st.title("Exploratory Data Analysis")
    EDA_choice = st.selectbox("Choose an operation of the following to perform:",['',
                                                                                    '1- Show shape',
                                                                                    '2- Show data types',
                                                                                    '3- Show missing values',
                                                                                    '4- Summary',
                                                                                    '5- Show columns',
                                                                                    '6- Show selected columns',
                                                                                    '7- Show Value Counts'])
    if EDA_choice == "1- Show shape":
        st.write(df.shape)
    elif EDA_choice == "2- Show data types": 
        df = df.astype(str)
        st.write(df.dtypes)
    elif EDA_choice == "3- Show missing values":  
        st.write(df.isna().sum())
    elif EDA_choice == "4- Summary":  
        st.write(df.info)
    elif EDA_choice == "5- Show columns":      
        st.write(df.columns)
    elif EDA_choice == "6- Show selected columns": 
        selected_columns = st.multiselect("Select the desired columns",df.columns)
        df = df[selected_columns]
        st.dataframe(df)
    elif EDA_choice == "7- Show Value Counts": 
        try:
            cols = st.multiselect("Select the column you want to show its value counts",df.columns)
            new_df = df[cols]
            st.write(new_df.value_counts().rename(index="Value"))
        except : 
            pass   

    # Data Visualization Part 
    plot_choice = st.selectbox("Select a specific type of plot : ",[
                                                                          '',
                                                                          '1- Box Plot',
                                                                          '2- Correlation Plot',
                                                                          '3- Pie Plot',
                                                                          '4- Scatter Plot',
                                                                          '5- Bar Plot'
                                                                        ]) 
    if plot_choice == "1- Box Plot" :
        col = st.selectbox("Choose only one column :",df.columns)
        fig = px.box(df, y = col)
        st.plotly_chart(fig)

    elif plot_choice == "2- Correlation Plot":
        corr_mat = df.corr()
        plt.figure(figsize=(30,25))
        fig = sns.heatmap(corr_mat, cmap = "coolwarm", annot = True, center = 0, fmt = ".2g")
        st.pyplot(plt)

    elif plot_choice == "3- Pie Plot":
        col = st.selectbox("Choose only one column :",df.columns)
        c_vc = df[col].value_counts()
        fig, ax = plt.subplots()
        ax.pie(c_vc, labels=c_vc.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.write(fig)

    elif plot_choice == "4- Scatter Plot":
        try :
            selected_columns = st.multiselect("Choose two columns :",df.columns)
            f_col = selected_columns[0]    
            sec_col = selected_columns[1]  
            fig = px.scatter(df, x=f_col, y=sec_col)
            fig.update_layout(title="Scatter Plot", xaxis_title=f_col, yaxis_title=sec_col) 
            st.plotly_chart(fig) 
        except :
            pass

    elif plot_choice == "5- Bar Plot":  
        try :
            selected_columns = st.multiselect("Choose two columns :",df.columns)
            f_col = selected_columns[0]    
            sec_col = selected_columns[1]    
            fig = px.bar(df, x=f_col, y=sec_col, title="Bar Plot")
            st.plotly_chart(fig)
        except :
            pass 


 
if choice == "Data Preparing & Modelling":
    st.title("Preparing the data before making the machine learning model")

    # Dropping unnecessary columns

    drop_or_not = st.selectbox("Do you want to drop any columns?.",["","Yes","No"])
    if drop_or_not == "No":
        st.warning("It is recommended to drop the unnecessary columns like Name, Cust_ID, etc.")
    if drop_or_not == "Yes":
        cols_to_drop = st.multiselect("Choose one or more than one column to drop :",df.columns)
        if cols_to_drop :
            df = df.drop(cols_to_drop, axis=1)  
            st.dataframe(df)  
            st.success("Columns were dropped successfully!")


    # Encoding some categorical columns 
    from sklearn.preprocessing import LabelEncoder
    att_atts = [""] + df.columns.tolist()
    att = st.selectbox("Choose the target feature :",att_atts)   
    try :
        if df[att].dtype == "object" or df[att].nunique() <= 1:
            lb = LabelEncoder()
            df[att] = lb.fit_transform(df[att])
    except :
        pass   

    
    ctg_cols = df.select_dtypes(include=['object', 'category'])
    if len(ctg_cols.columns) > 1 :

        one_hot_enc = pd.get_dummies(ctg_cols, drop_first=True)
        df_enc = pd.concat([df.drop(columns=ctg_cols), one_hot_enc], axis=1)

    else :
        df_enc = df

    fill_miss_ctg = st.selectbox('In What way do you prefer to handle your missing categorical data?',
                                            ['', 'Most Frequent', 'Additional Class'])

    try :
        if fill_miss_ctg == 'Most Frequent':
            df_enc[ctg_cols.columns] = df_enc[ctg_cols.columns].fillna(
                df_enc[ctg_cols.columns].mode().iloc[0])

        if fill_miss_ctg == 'Additional Class':
            df_enc[ctg_cols.columns] = df_enc[ctg_cols.columns].fillna('Missing')

    except :
        pass


    fill_miss = st.selectbox('In What way do you prefer to handle your missing continous data ?',['','Mean','Median','Mode'])
    cont_cols = df_enc.select_dtypes(include=['float64', 'int64'])
    if fill_miss == 'Mean':
        df_enc[cont_cols.columns] = df_enc[cont_cols.columns].fillna(df_enc[cont_cols.columns].mean())

    if fill_miss == 'Median' :
        df_enc[cont_cols.columns] = df_enc[cont_cols.columns].fillna(df_enc[cont_cols.columns].median())

    if fill_miss == 'Mode' :
        df_enc[cont_cols.columns] = df_enc[cont_cols.columns].fillna(df_enc[cont_cols.columns].iloc[0])
        
    from sklearn.preprocessing import MinMaxScaler
    try:
        X = df_enc.drop(columns=att)
        y = df_enc[att]
        st.write('Your Features are', X)
        st.write('Your Target is', y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        mm = MinMaxScaler()
        scaled_data = mm.fit_transform(X)
        df_scaled = pd.DataFrame(scaled_data, columns=X.columns)
        df_scaled['target'] = y
        st.write('your data after preprocessing', df_scaled)

    except:
        pass


    try :
        if y.dtype == 'object' or y.nunique() <= 10:
            st.info('This is a classification problem')
            modeling_choice = st.selectbox('Do you want Auto modeling or you want to choose the model ?',
                                           ['', 'Auto modeling', 'Manual modeling'])

            if modeling_choice == 'Auto modeling':
                from pycaret.classification import *

                if st.button('Run Modelling'):
                    setup(df, target=att, verbose=False)
                    setup_df = pull()
                    st.info("This is the ML process")
                    st.dataframe(setup_df)
                    st.error('Wait for a few seconds. It will take a little bit of time.')
                    best_model = compare_models(include=['lr','dt','rf','knn','nb'])
                    compare_df = pull()
                    st.info("This is your ML model")
                    st.dataframe(compare_df)
                    save_model(best_model, 'best_model')

                    with open('best_model.pkl', 'rb') as model_file:
                        st.download_button('Download the model', model_file, 'best_model.pkl')

            if modeling_choice == 'Manual modeling':

                algo_type = st.selectbox('Please choose which type of algorithm you want to use',
                                         ['', 'Logistic Regression', 'Decision Trees', 'Random Forest', 'SVC',
                                          'KNN'])

                if algo_type == 'Logistic Regression':
                    from sklearn.linear_model import LogisticRegression

                    clf = LogisticRegression(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'Decision Trees':
                    from sklearn.tree import DecisionTreeClassifier

                    clf = DecisionTreeClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier

                    clf = RandomForestClassifier(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'SVC':
                    from sklearn.svm import SVC

                    clf = SVC(random_state=42)
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                if algo_type == 'KNN':
                    from sklearn.neighbors import KNeighborsClassifier

                    clf = KNeighborsClassifier()
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                evaluation_type = st.selectbox('Choose type of evaluation metrics ', ['', 'Accuracy', 'Confusion Matrix',
                                                                                      'Precision, Recall, and F1-score'])

                if evaluation_type == 'Accuracy':
                    from sklearn.metrics import accuracy_score

                    accuracy = accuracy_score(y_test, y_pred)
                    st.write("Accuracy:", accuracy)

                if evaluation_type == 'Confusion Matrix':
                    from sklearn.metrics import confusion_matrix

                    cm = confusion_matrix(y_test, y_pred)
                    st.write("Confusion Matrix:")
                    st.dataframe(cm)

                if evaluation_type == 'Precision, Recall, and F1-score':
                    from sklearn.metrics import precision_score, recall_score, f1_score

                    precision = precision_score(y_test, y_pred, average='macro')
                    recall = recall_score(y_test, y_pred, average='macro')
                    f1 = f1_score(y_test, y_pred, average='macro')
                    metrics_dict = {
                        "Metric": ["Precision", "Recall", "F1-Score"],
                        "Value": [precision, recall, f1]
                    }
                    metrics_df = pd.DataFrame(metrics_dict)
                    st.dataframe(metrics_df)

                try:
                    import pickle
                    model_filename = "clf.pkl"
                    with open(model_filename, "wb") as model_file:
                        pickle.dump(clf, model_file)

                    st.download_button('Download the model', open(model_filename, 'rb').read(), 'clf.pkl')

                except:
                    pass



        else:
            st.info('This is a regression problem')
            modeling_choice = st.selectbox('Do you want Auto modeling or you want to choose the model ?',
                                           ['', 'Auto modeling', 'Manual modeling'])

            if modeling_choice == 'Auto modeling':

                from pycaret.regression import *

                if st.button('Run Modelling'):
                    setup(df, target=att, verbose=False)
                    setup_df = pull()
                    st.info("This is the ML process")
                    st.dataframe(setup_df)
                    st.error('Wait for a few seconds. It will take a little bit of time.')
                    best_model = compare_models(include=['lr','ridge','lasso','rf','en'])
                    compare_df = pull()
                    st.info("This is your ML model")
                    st.dataframe(compare_df)
                    save_model(best_model, 'best_model')

                    with open('best_model.pkl', 'rb') as model_file:
                        st.download_button('Download the model', model_file, 'best_model.pkl')

            if modeling_choice == 'Manual modeling':

                algo_type = st.selectbox('Please choose which type of algorithm do you want to use ?',
                                         ['', 'Linear Regression', 'Ridge', 'SVR', 'Random Forest'])

                if algo_type == 'Linear Regression':
                    from sklearn.linear_model import LinearRegression

                    rg = LinearRegression()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                if algo_type == 'Ridge':
                    from sklearn.linear_model import Ridge

                    rg = Ridge()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                if algo_type == 'SVR':
                    from sklearn.svm import SVR

                    rg = SVR()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                if algo_type == 'Random Forest':
                    from sklearn.ensemble import RandomForestRegressor

                    rg = RandomForestRegressor()
                    rg.fit(X_train, y_train)
                    y_pred = rg.predict(X_test)

                evaluation_type = st.selectbox('Choose type of evaluation metrics ', ['', 'MAE', 'MSE', 'r2 score'])

                if evaluation_type == 'MAE':
                    from sklearn.metrics import mean_absolute_error

                    MAE = mean_absolute_error(y_test, y_pred)
                    st.write("Mean absolute error:", MAE)

                if evaluation_type == 'MSE':
                    from sklearn.metrics import mean_squared_error

                    MSE = mean_squared_error(y_test, y_pred)
                    st.write("Mean squared error:", MSE)

                if evaluation_type == 'r2 score':
                    from sklearn.metrics import r2_score

                    r2 = r2_score(y_test, y_pred)
                    st.write("r2 score:", r2)

                try:

                    model_filename = "rg.pkl"
                    with open(model_filename, "wb") as model_file:
                        pickle.dump(rg, model_file)

                    st.download_button('Download the model', open(model_filename, 'rb').read(), 'rg.pkl')

                except:
                    pass

    except :
        pass
