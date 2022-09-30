from config.config import params
from data.dal.solar_ftp_dal import SolarFTPDAO
from services.similarity.cosine_similarity import CosineSimilarity
from services.analysis.analysis import  Analysis

ENV = 'PROD'
print(params[ENV]['solar_url'])

solar_ftp = SolarFTPDAO(ENV)
solar_data = solar_ftp.get_solar_data()
solar_years = list(solar_data.index.values)
print(solar_data.head())
cs_similarity = CosineSimilarity()
similarity_matrix = cs_similarity.similarity(solar_data)
analysis_svc = Analysis()
top_n_years = analysis_svc.top_n_similar_pairs(similarity_matrix, 5, solar_years)
print('Top 5 Similar Years: {0}'.format(top_n_years))
most_like_2012 = analysis_svc.pair_most_like_given(similarity_matrix, 2012, solar_years)
print('Most like 2012: {0}'.format(most_like_2012))

