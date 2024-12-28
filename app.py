import streamlit as st
import pandas as pd
import preprocessor, helper
import plotly.express as px
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy
import plotly.figure_factory as ff
df = pd.read_csv('athlete_events.csv')
region_df = pd.read_csv('noc_regions.csv')

# Preprocess the data
df = preprocessor.preprocess(df,region_df)
st.sidebar.title('Olympics Analysis')
st.sidebar.image('data:image/png;base64,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')
# Sidebar menu
user_menu = st.sidebar.radio(
    'Select an option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete wise Analysis')
)

# Display the preprocessed data


# Handle the selected menu option
if user_menu == 'Medal Tally':  # Match the option name correctly
    st.sidebar.header('Medal Tally')
    years, country =helper.country_year_list(df)
    selected_year = st.sidebar.selectbox("Select Year",years)
    selected_country = st.sidebar.selectbox("Select Country", country)
    medal_tally = helper.fetch_medal_tally(df,selected_year,selected_country)  # Pass the DataFrame as an argument
    if selected_year == "Overall" and selected_country == "Overall":
        st.title("Overall Tally")
    if selected_year != "Overall" and selected_country == "Overall":
        st.title("Medal tally in " + str(selected_year) + " Olympics")

    if selected_year == "Overall" and selected_country != "Overall":
        st.title(selected_country+ " overall performance")
    if selected_year != "Overall" and selected_country != "Overall":
        st.title(selected_country + " overall performance in "+ str(selected_year) + " Olympics")

    st.table(medal_tally)
if user_menu == 'Overall Analysis':
    editions = df['Year'].unique().shape[0] - 1
    cities = df['City'].unique().shape[0]
    sports = df['Sport'].unique().shape[0]
    events = df['Event'].unique().shape[0]
    athletes = df['Name'].unique().shape[0]
    nations = df['region'].unique().shape[0]
    st.title("Top Statistics")
    col1,col2,col3 = st.columns(3)
    with col1 :
        st.header("Editions")
        st.title(editions)
    with col2 :
        st.header("Hosts")
        st.title(cities)
    with col3 :
        st.header("Sports")
        st.title(sports)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Events")
        st.title(events)
    with col2:
        st.header("Nations")
        st.title(nations)
    with col3:
        st.header("Athletes")
        st.title(athletes)
        # Participating Nations Over Time
    nations_over_time = helper.data_over_time(df, 'region')
    fig1 = px.line(nations_over_time, x="Edition", y="region")
    st.title("Participating Nations Over the Years")
    st.plotly_chart(fig1)

    # Events Over Time
    events_over_time = helper.data_over_time(df, 'Event')
    fig2 = px.line(events_over_time, x="Edition", y="Event")
    st.title("Events Over the Years")
    st.plotly_chart(fig2)
    athlete_over_time = helper.data_over_time(df, 'Name')
    fig2 = px.line(athlete_over_time, x="Edition", y="Name")
    st.title("Events Over the Years")
    st.plotly_chart(fig2)
    st.title("No. of Events over time(Every Sport)")
    fig,ax = plt.subplots(figsize = (20,20))
    x = df.drop_duplicates(['Year', 'Sport', 'Event'])
    ax = sns.heatmap(x.pivot_table(index='Sport', columns='Year', values='Event', aggfunc='count').fillna(0).astype('int'),
                annot=True)
    st.pyplot(fig)
    st.title("Most successful Athletes")
    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0,'Overall')
    selected_sport = st.selectbox('Select a Sport',sport_list)
    x = helper.most_successful(df,selected_sport)
    st.table(x)

if user_menu == 'Country-wise Analysis':

    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country', country_list )




    country_df = helper.yearwise_medal_tally(df,selected_country)
    fig = px.line(country_df, x="Year", y="Medal")
    st.title(selected_country + " Medal Tally over the Years")
    st.plotly_chart(fig)

    st.title(selected_country + " excel in the following sports")
    pt = helper.country_event_heatmap(df , selected_country)

    fig, ax = plt.subplots(figsize=(20, 20))

    ax = sns.heatmap(pt,
        annot=True)
    st.pyplot(fig)

    st.title("Top 10 athletes of " + selected_country)
    top10_df = helper.most_successful_countrywise(df , selected_country)
    st.table(top10_df)





if user_menu == 'Athlete wise Analysis':
    # Remove duplicates based on Name and region
    athlete_df = df.drop_duplicates(subset=['Name', 'region'])

    # Age for all athletes
    x1 = athlete_df['Age'].dropna()

    # Age for athletes who won Gold, Silver, and Bronze
    x2 = athlete_df[athlete_df['Medal'] == 'Gold']['Age'].dropna()
    x3 = athlete_df[athlete_df['Medal'] == 'Silver']['Age'].dropna()
    x4 = athlete_df[athlete_df['Medal'] == 'Bronze']['Age'].dropna()

    # Create a distribution plot
    fig = ff.create_distplot([x1, x2, x3, x4],
                             ['Overall Age', 'Gold Medalist', 'Silver Medalist', 'Bronze Medalist'],
                             show_hist=False, show_rug=False)

    # Display the plot in Streamlit
    fig.update_layout(autosize = False , width =1000, height =600)
    st.title("Distribution of Age")
    st.plotly_chart(fig)

    x = []
    name = []

    # List of famous sports
    famous_sports = [
        'Basketball', 'Judo', 'Football', 'Tug-Of-War', 'Athletics',
        'Swimming', 'Badminton', 'Sailing', 'Gymnastics',
        'Art Competitions', 'Handball', 'Weightlifting', 'Wrestling',
        'Water Polo', 'Hockey', 'Rowing', 'Fencing',
        'Shooting', 'Boxing', 'Taekwondo', 'Cycling', 'Diving', 'Canoeing',
        'Tennis', 'Golf', 'Softball', 'Archery',
        'Volleyball', 'Synchronized Swimming', 'Table Tennis', 'Baseball',
        'Rhythmic Gymnastics', 'Rugby Sevens', 'Rugby', 'Polo', 'Ice Hockey'
    ]

    # Iterate over the sports and process data
    for sport in famous_sports:
        temp_df = athlete_df[athlete_df['Sport'] == sport]
        x.append(temp_df[temp_df['Medal'] == 'Gold']['Age'].dropna())
        name.append(sport)

    # Create distribution plot
    fig = ff.create_distplot(x, name, show_hist=False, show_rug=False)

    # Display the plot in Streamlit
    fig.update_layout(autosize=False, width=1000, height=600)
    st.title("Distribution of Age wrt Sports(Gold Medalist)")
    st.plotly_chart(fig)


    sport_list = df['Sport'].unique().tolist()
    sport_list.sort()
    sport_list.insert(0, 'Overall')

    st.title('Height vs Weight')
    selected_sport = st.selectbox('Select a Sport', sport_list)



    temp_df = helper.weight_v_height (df,selected_sport)
    fig,ax = plt.subplots()
    ax = sns.scatterplot(x='Weight', y='Height', data= temp_df, hue = temp_df['Medal'] , style = temp_df['Sex'], s = 60)


    st.pyplot(fig)

    st.title("Men vs Women participation over the years")
    final = helper.men_vs_women(df)
    fig = px.line(final, x='Year', y=['Male', 'Female'])
    fig.update_layout(autosize=False, width=1000, height=600)
    st.plotly_chart(fig)