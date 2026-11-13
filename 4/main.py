import warnings
import matplotlib.pyplot as plt
import real_ts as rt
import synthetic_ts as st
import superstore as ss

warnings.filterwarnings('ignore')

# Частина 1: офіційний курс долара (НБУ)
print("========== ГРУПА ВИМОГ 1: офіційний курс USD (НБУ) ==========")
ts = rt.load_nbu('data/usd_2023-2024_1.xlsx')
rt.plot_series(ts)
rt.print_properties('Властивості реального ряду', rt.ts_properties(ts.values))
rt.stl_decomposition(ts)
rt.correlation_analysis(ts)
rt.week_clustering(ts)

params = st.fit_model(ts)
st.compare(ts, params)

# Частина 2: продажі магазину Superstore (Data_Set_10)
print("\n========== ГРУПА ВИМОГ 2: Data_Set_10 (Superstore) ==========")
orders, monthly = ss.load_superstore('data/Data_Set_10.xls')
res = ss.properties_and_decomposition(monthly)
ss.correlation(orders, monthly)
ss.subcategory_clustering(orders)
ss.synthetic(monthly, res)

plt.show()
