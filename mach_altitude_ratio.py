# %%

from traffic.data.datasets import paris_toulouse_2017
import altair as alt

# %%

subset = paris_toulouse_2017.between("2017-10-09", "2017-10-16").phases().eval()
df = subset.data[subset.data.phase == "CRUISE"]

# %%

altitudes = df.altitude
groundspeeds = df.groundspeed
# %%

alt.Chart(df).mark_circle(size=60).encode(
    x="altitude",
    y="groundspeed",
    color="flight_id",
).interactive()
# %%
