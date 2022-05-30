import pandas as pd
import streamlit as st
import datetime
import pickle
# monthdic = {
#     'Jan': 1, 'Feb': 2, 'Mar': 2, 'Apr': 2, 'May': 2, 'Jun': 2, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
# }
# st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("MARKETING SPEND PREDICTION MODEL")
# st.subheader('Weekly')
st.write("")
st.write("Please enter values with respect to monthly data ...")

orders2 = st.number_input(
    label="Monthly orders", step=1., format="%.2f")
st.write('You entered orders:', orders2)

revenue2 = st.number_input(
    label="Monthly revenue", step=1., format="%.2f")
st.write('You entered revenue:', revenue2)

averagePrice2 = st.number_input(
    label="Monthly average price", step=1., format="%.2f")
st.write('You entered Average Price:', averagePrice2)

facebookPurchases2 = st.number_input(
    label="Monthly facebook purchases", step=1., format="%.2f")
st.write('You entered facebook purchases:', facebookPurchases2)

facebookRevenue2 = st.number_input(
    label="Monthly facebook revenue", step=1., format="%.2f")
st.write('You entered Facebook Revenue:', facebookRevenue2)

googleSpend2 = st.number_input(
    label="Monthly google spend", step=1., format="%.2f")
st.write('You entered google spend:', googleSpend2)

googlePurchases2 = st.number_input(
    label="Monthly google purchases", step=1., format="%.2f")
st.write('You entered google purchases:', googlePurchases2)

googleRevenue2 = st.number_input(
    label="Monthly google revenue", step=1., format="%.2f")
st.write('You entered google revenue:', googleRevenue2)

facebookROAS2 = st.number_input(
    label="Monthly facebook ROAS", step=1., format="%.2f")
st.write('You entered facebook ROAS:', facebookROAS2)

googleCPA2 = st.number_input(
    label="Monthly google CPA", step=1., format="%.2f")
st.write('You entered google CPA:', googleCPA2)

googleROAS2 = st.number_input(
    label="Monthly google ROAS", step=1., format="%.2f")
st.write('You entered google ROAS:', googleROAS2)

totalROAS2 = st.number_input(
    label="Monthly total ROAS", step=1., format="%.2f")
st.write('You entered total ROAS:', totalROAS2)

targetRevenue2 = st.number_input(
    label="Monthly target revenue", step=1., format="%.2f")
st.write('You entered target revenue:', targetRevenue2)

#
# month = monthdic[Month]
# year = 2021
# Week_Number = #int(datetime.date(year,month , 1).strftime("%V"))
# M_ActualRevenue = targetRevenue  # (targetRevenue/20)*7


if orders2 > 0 and revenue2 > 0 and averagePrice2 > 0 and facebookPurchases2 > 0 and facebookRevenue2 > 0 and googleSpend2 > 0 and googlePurchases2 > 0 and googleRevenue2 > 0 and facebookROAS2 > 0 and googleCPA2 > 0 and googleROAS2 > 0 and totalROAS2 > 0 and targetRevenue2 > 0:
    st.write('----------------------------------- Marketing Spend Budget Prediction ----------------------------------------')

    # ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend', 'ga_purchases', 'ga_revenue', 'fb_roas','ga_roas', 'ga_cpa', 'roas', 'target_revenue']

    test2 = {'orders': orders2, 'revenue': revenue2,
             'average_price': averagePrice2, 'fb_purchases': facebookPurchases2, 'fb_revenue': facebookRevenue2,
             'ga_spend': googleSpend2, 'ga_purchases': googlePurchases2, 'ga_revenue': googleRevenue2,
             'fb_roas': facebookROAS2, 'ga_cpa': googleCPA2, 'ga_roas': googleROAS2, 'roas': totalROAS2, 'target_revenue': targetRevenue2}

    test_DF2 = pd.DataFrame([test2])

    # path2 = '/./model_scaler/'
    inner2 = pickle.load(open('./model_scaler/inner_1.gz', 'rb'))
    outer2 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model2 = pickle.load(open('./model_scaler/xgb_reg_1.pkl', "rb"))

    test_DF2 = pd.DataFrame(inner2.transform(test_DF2))
    test_DF2 = test_DF2.rename(columns={0: 'orders', 1: 'revenue', 2: 'average_price', 3: 'fb_purchases',
                                        4: 'fb_revenue', 5: 'ga_spend', 6: 'ga_purchases', 7: 'ga_revenue', 8: 'fb_roas', 9: 'ga_cpa', 10: 'ga_roas', 11: 'roas', 12: 'target_revenue'})

    pred2 = model2.predict(test_DF2)
    pred2 = pred2.reshape(-1, 1)
    pred2 = outer2.inverse_transform(pred2)
    totalFB_Revenue2 = float(pred2[0])
    totalGA_Revenue2 = float(totalFB_Revenue2*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue2)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue2)

    test2 = {'orders': orders2, 'revenue': revenue2,
             'average_price': averagePrice2, 'fb_purchases': facebookPurchases2, 'fb_revenue': facebookRevenue2,
             'ga_spend': googleSpend2, 'ga_purchases': googlePurchases2, 'ga_revenue': googleRevenue2,
             'fb_roas': facebookROAS2, 'ga_cpa': googleCPA2, 'ga_roas': googleROAS2, 'roas': totalROAS2}

    test_DF2 = pd.DataFrame([test2])

    # path2 = '/./model_scaler/'
    inner2 = pickle.load(open('./model_scaler/inner_2.gz', 'rb'))
    outer2 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model2 = pickle.load(open('./model_scaler/xgb_reg_2.pkl', "rb"))

    test_DF2 = pd.DataFrame(inner2.transform(test_DF2))

    test_DF2 = test_DF2.rename(columns={0: 'orders', 1: 'revenue', 2: 'average_price', 3: 'fb_purchases',
                                        4: 'fb_revenue', 5: 'ga_spend', 6: 'ga_purchases', 7: 'ga_revenue', 8: 'fb_roas', 9: 'ga_cpa', 10: 'ga_roas', 11: 'roas'})

    pred2 = model2.predict(test_DF2)
    pred2 = pred2.reshape(-1, 1)
    pred2 = outer2.inverse_transform(pred2)
    totalFB_Revenue2 = float(pred2[0])
    totalGA_Revenue2 = float(totalFB_Revenue2*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue2)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue2)

    test2 = {'orders': orders2, 'revenue': revenue2,
             'average_price': averagePrice2, 'fb_purchases': facebookPurchases2, 'fb_revenue': facebookRevenue2,
             'ga_spend': googleSpend2, 'ga_purchases': googlePurchases2, 'ga_revenue': googleRevenue2,
             'fb_roas': facebookROAS2, 'ga_roas': googleROAS2, 'roas': totalROAS2}

    test_DF2 = pd.DataFrame([test2])

    # path2 = '/./model_scaler/'
    inner2 = pickle.load(open('./model_scaler/inner_2.gz', 'rb'))
    outer2 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model2 = pickle.load(open('./model_scaler/xgb_reg_3.pkl', "rb"))

    test_DF2 = pd.DataFrame(inner2.transform(test_DF2))

    test_DF2 = test_DF2.rename(columns={0: 'orders', 1: 'revenue', 2: 'average_price', 3: 'fb_purchases',
                                        4: 'fb_revenue', 5: 'ga_spend', 6: 'ga_purchases', 7: 'ga_revenue', 8: 'fb_roas', 9: 'ga_roas', 10: 'roas'})

    pred2 = model2.predict(test_DF2)
    pred2 = pred2.reshape(-1, 1)
    pred2 = outer2.inverse_transform(pred2)
    totalFB_Revenue2 = float(pred2[0])
    totalGA_Revenue2 = float(totalFB_Revenue2*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue2)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue2)

    test2 = {'revenue': revenue2,
             'average_price': averagePrice2, 'fb_purchases': facebookPurchases2, 'fb_revenue': facebookRevenue2,
             'ga_spend': googleSpend2, 'ga_purchases': googlePurchases2, 'ga_revenue': googleRevenue2,
             'fb_roas': facebookROAS2, 'ga_roas': googleROAS2, 'roas': totalROAS2}

    test_DF22 = pd.DataFrame([test2])

    # path2 = '/./model_scaler/'
    inner2 = pickle.load(open('./model_scaler/inner_2.gz', 'rb'))
    outer2 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model2 = pickle.load(open('./model_scaler/xgb_reg_4.pkl', "rb"))

    test_DF2 = pd.DataFrame(inner2.transform(test_DF22))

    test_DF2 = test_DF2.rename(columns={0: 'revenue', 1: 'average_price', 2: 'fb_purchases',
                                        3: 'fb_revenue', 4: 'ga_spend', 5: 'ga_purchases', 6: 'ga_revenue', 7: 'fb_roas', 8: 'ga_roas', 9: 'roas'})

    pred2 = model2.predict(test_DF2)
    pred2 = pred2.reshape(-1, 1)
    pred2 = outer2.inverse_transform(pred2)
    totalFB_Revenue2 = float(pred2[0])
    totalGA_Revenue2 = float(totalFB_Revenue2*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue2)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue2)

    test2 = {'revenue': revenue2,
             'average_price': averagePrice2, 'fb_purchases': facebookPurchases2, 'fb_revenue': facebookRevenue2,
             'ga_spend': googleSpend2, 'ga_purchases': googlePurchases2,
             'fb_roas': facebookROAS2, 'ga_roas': googleROAS2, 'roas': totalROAS2}

    test_DF2 = pd.DataFrame([test2])

    # path2 = '/./model_scaler/'
    inner2 = pickle.load(open('./model_scaler/inner_2.gz', 'rb'))
    outer2 = pickle.load(open('./model_scaler/outer.gz', 'rb'))

    model2 = pickle.load(open('./model_scaler/xgb_reg_5.pkl', "rb"))

    test_DF2 = pd.DataFrame(inner2.transform(test_DF2))

    test_DF2 = test_DF2.rename(columns={0: 'revenue', 1: 'average_price', 2: 'fb_purchases',
                                        2: 'fb_revenue', 2: 'ga_spend', 2: 'ga_purchases', 2: 'fb_roas', 7: 'ga_roas', 8: 'roas'})

    pred2 = model2.predict(test_DF2)
    pred2 = pred2.reshape(-1, 1)
    pred2 = outer2.inverse_transform(pred2)
    totalFB_Revenue2 = float(pred2[0])
    totalGA_Revenue2 = float(totalFB_Revenue2*0.1)

    st.write('Facebook Marketing Spend Budget :', totalFB_Revenue2)
    st.write('Google Marketing Spend Budget :', totalGA_Revenue2)


# ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend',
#     'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_cpa', 'ga_roas', 'roas', 'target_revenue']
# ['orders', 'revenue', 'average_price', 'fb_purchases', 'fb_revenue', 'ga_spend',
#     'ga_purchases', 'ga_revenue', 'fb_roas', 'ga_roas', 'ga_cpa', 'roas', 'target_revenue']
