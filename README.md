# competition_test

### インストール関係

condaを使う場合

```conda install -c conda-forge fbprophet```

condaを使わない場合

```pip install fbprophet```

sample code(サイトより抜粋)
```
import pandas as pd
from fbprophet import Prophet

df = pd.read_csv('../dat/受注予測データ/受注予測データ.csv')
df.head()

dat = df.groupby("date",as_index=False).sum()

#データフレーム整形
dat = dat.loc[:,["date","amount"]]

#リネーム、prophet用に定義する
columns={
    "date":"ds",
    "amount":"y"
}
#再度読み込みを行う
dat.rename(columns=columns, inplace=True)

dat.head()

dat.tail()

#予測モデル作成・フィッティング
m = Prophet()
m.fit(dat)

#予測モデル作成・フィッティング
m = Prophet()
m.fit(dat)

#データフレーム内各カラムを確認
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#描画
#時系列
fig1 = m.plot(forecast)

#描画
#構成要素
fig2 = m.plot_components(forecast)

from fbprophet.plot import add_changepoints_to_plot
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

### default changepoint_prior_scale=0.05
m = Prophet(changepoint_prior_scale=0.5)
forecast = m.fit(dat).predict(future)
fig = m.plot(forecast)

### default changepoint_prior_scale=0.05
m = Prophet(changepoint_prior_scale=0.001)
forecast = m.fit(dat).predict(future)
fig = m.plot(forecast)

m = Prophet(seasonality_mode='multiplicative', mcmc_samples=300).fit(dat)
fcst = m.predict(future)
fig = m.plot_components(fcst)
