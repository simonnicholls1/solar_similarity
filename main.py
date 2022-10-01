from config.config import params
from data.dal.solar_ftp_dao import SolarFTPDAO
from services.similarity.cosine_similarity import CosineSimilarity
from services.similarity.euclidean_similarity import EuclideanSimilarity
from services.similarity.corr_similarity import CorrelationSimilarity
from services.analysis.analysis import Analysis
import logging
import matplotlib.pyplot as plt

ENV = 'PROD'

#Loading of data
print(params[ENV]['solar_url'])
solar_ftp = SolarFTPDAO(ENV)
solar_data = solar_ftp.solar_data
solar_years = list(solar_data.index.values)
print(solar_data.head())

#Setup Analysis class
analysis_svc = Analysis()

#Setup Similarity Measure Services
cs_similarity = CosineSimilarity()
corr_similarity = CorrelationSimilarity()
ec_similarity = EuclideanSimilarity()

#Cosine similarity
similarity_matrix = cs_similarity.similarity(solar_data)
top_n_years_cosine = analysis_svc.top_n_similar_pairs(similarity_matrix, 5, solar_years, True)
most_like_2012_cosine = analysis_svc.pair_most_like_given(similarity_matrix, 2012, solar_years, True)
print('Cosine Similarity - Top 5 Similar Years: {0}'.format(top_n_years_cosine))
print('Cosine Similarity - Most like 2012: {0}'.format(most_like_2012_cosine))

#Correlation Similarity
similarity_matrix = corr_similarity.similarity(solar_data)
top_n_years_corr = analysis_svc.top_n_similar_pairs(similarity_matrix, 5, solar_years, True)
most_like_2012_corr = analysis_svc.pair_most_like_given(similarity_matrix, 2012, solar_years, True)
print('Correlation Similarity - Top 5 Similar Years: {0}'.format(top_n_years_corr))
print('Correlation Similarity - Most like 2012: {0}'.format(most_like_2012_corr))

#Euclidean Similarity
similarity_matrix = ec_similarity.similarity(solar_data)
top_n_years_ec = analysis_svc.top_n_similar_pairs(similarity_matrix, 5, solar_years, False)
most_like_2012_ec = analysis_svc.pair_most_like_given(similarity_matrix, 2012, solar_years, False)
print('Euclidean Similarity - Top 5 Similar Years: {0}'.format(top_n_years_ec))
print('Euclidean Similarity - Most like 2012: {0}'.format(most_like_2012_ec))

# Plots

# Plot the solar data by month - here we can see large seasonal data roughly every ten years
fig1 = plt.figure("Solar Data Monthly")
plt.plot(solar_data)
plt.xlabel("Year")
plt.ylabel("Solar Radiation")
plt.title("Solar Data Monthly")

# Plotting each year doesn't tell us much as there are a lot of values.
fig2 = plt.figure("Solar Data Yearly")
plt.plot(solar_data.transpose())
plt.xlabel("Month")
plt.ylabel("Solar Radiation")
plt.title("Solar Data Yearly")

# Plot the different closest years to 2012 for each similarity measure
# Can see that the euclidean measure takes into account magnitude
# Whereas the others do not, the rely more on following the same pattern
similar_2012 = [2012, most_like_2012_ec, most_like_2012_corr, most_like_2012_cosine]
fig3 = plt.figure("Closest Year for each Similarity Measure")
plt.plot(solar_data.loc[list(similar_2012), :].transpose())
plt.xlabel("Month")
plt.ylabel("Solar Radiation")
plt.legend(['2012', 'Euclidean', 'Correlation', 'Cosine'], loc='upper left')
plt.title("Closest Year for each Similarity Measure")

# Plot the top pair for Euclidean similarity
fig4 = plt.figure("Top Pair for Euclidean Similarity")
plt.plot(solar_data.loc[[top_n_years_ec[0][0], top_n_years_ec[0][1]], :].transpose())
plt.xlabel("Month")
plt.ylabel("Solar Radiation")
plt.legend([top_n_years_ec[0][0], top_n_years_ec[0][1]], loc='upper left')
plt.title("Top Pair for Euclidean Similarity")

# Plot the top pair for Cosine Similarity
fig5 = plt.figure("Top Pair for Cosine Similarity")
plt.plot(solar_data.loc[[top_n_years_cosine[0][0], top_n_years_cosine[0][1]], :].transpose())
plt.xlabel("Month")
plt.ylabel("Solar Radiation")
plt.legend([top_n_years_cosine[0][0], top_n_years_cosine[0][1]], loc='upper left')
plt.title("Top Pair for Cosine Similarity")

# Plot scatter for top pair from Correlation Similarity
fig5 = plt.figure("Scatter plot for Top Pair")
x = solar_data.loc[top_n_years_corr[0][0]]
y = solar_data.loc[top_n_years_corr[0][1]]
plt.scatter(x, y)
plt.xlabel(str(top_n_years_cosine[0][0]))
plt.ylabel(str(top_n_years_cosine[0][1]))
plt.title("Scatter plot for Top Pair from Correlation Similarity")

# Delta between years for top 5 pairs
delta_years = [abs(x[0]-x[1]) for x in top_n_years_cosine]

print('Finished Script')






