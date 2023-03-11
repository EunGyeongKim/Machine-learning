import pandas as pd

results = pd.read_csv("/content/data.csv")

tlist =['responsibilities', 'requirements', 'preference', 'Benefits']
# 빈 컬럼 만들어주기
for i in tlist:
    results[i]=''

# '주요업무', '자격요건', '우대사항', '혜택 및 복지'로 데이터 분리
for col in results:
    for row in range(len(results)):
        test = results['jd'][row]
        tmp =[]
        for i in ['주요업무', '자격요건', '우대사항', '혜택 및 복지']:
            tt = str(test).split(i)
            tmp.append(tt[0])
            if len(tt) < 2:
                test = "".join(tt)
            else:
                test = tt[1]
        tmp.append(test)

        for a, i in enumerate(tlist):
            results[i][row] = tmp[a+1]

# 타겟이 되는 직무 연자 컬럼 만들어주기
results['target_annual']=''


#최대 20년까지
for row in range(len(results)):
    results['target_annual'][row] = [i for i in range(results['annual_from'][row], min(results['annual_to'][row]+1, 20) )]

# 연차 필터링. 1~3년치만 출력
# results = results[results['target_annual'].astype(str).str.contains('1|2|3|')]  

!pip install krwordrank

#kwordrank 소개
#보통 이런 자연어 전처리를 할 때에는 konlpy를 사용하지만 그건 설치 방법이 조금 까다롭다. 그래서 우리의 목적인 워드클라우드를 만들어주기 위한 것만 달성하기 위해 krwordrank 사용
#기회가 된다면 konlpy 사용해보시는 것도 추천.

#kwordrank 자세한 설명은 https://lovit.github.io/nlp/2018/04/16/krwordrank/ 참고

import pandas as pd
import matplotlib.pyplot as plt
from krwordrank.word import KRWordRank
import numpy as np

df = results.copy() #복사

df['responsibilities'] = df['responsibilities'].replace(np.nan, '없음') 
#df['responsibilities'] = df['responsibilities'].str.replace('파이썬', 'Python')
df['requirements'] = df['requirements'].replace(np.nan, '없음')
#df['requirements'] = df['requirements'].str.replace('파이썬', 'Python')
df['preference'] = df['preference'].replace(np.nan, '없음')
#df['preference'] = df['preference'].str.replace('파이썬', 'Python')
df['Benefits'] = df['Benefits'].replace(np.nan, '없음')

#df = df[~df['target_annual'].astype(str).str.contains('5|7|8|9')]

df.head(5)

texts = df["responsibilities"].values.tolist() #주요업무의 값들을 리스트로 변화시켜줍니다. 그렇게 해서 모든 값들을 하나의 뭉탱이로 만든다
print(texts)


!pip install konlpy
from konlpy.tag import Okt
# kwordrank만 사용하여 출력했을 경우 형용사, 동사 등 쓸모없는 단어또한 같이 출력됨
# 그러므로 명사 단어만 사용하기 위해 knolpy 사용

tmp = []
okt = Okt()

# 각 문장에서 명사만 추출해 저장
for i in texts:
    tt = " ".join(okt.nouns(i))
    tmp.append(tt)

wordrank_extractor = KRWordRank(
    min_count = 3, # 단어의 최소 출현 빈도수 (그래프 생성 시)
    max_length = 15, # 단어의 최대 길이
    verbose = True
    )

beta = 0.85    # PageRank의 decaying factor beta
max_iter = 10

#단어추출기에 넣으면 키워드랑, 랭크랑, 그래프를 output으로 뱉어줌
keywords, rank, graph = wordrank_extractor.extract(tmp, beta, max_iter)  

stopwords = {'대한','분이면', '있습니다.','분석','데이터', '위한', 'and', '통해', '통한','있는','the','to','in','for','of',
             '다양한', '이를', '결과를', '위해', '필요한', 'on', '함께'}  #걸렀으면 하는 stopwords
passwords = {word:score for word, score in sorted(   
keywords.items(), key=lambda x:-x[1])[:100] if not (word in stopwords)}  #stopwords는 제외된 keywords 탑 300개

for word, r in sorted(passwords.items(), key=lambda x:x[1], reverse=True)[:30]: #top 30만 뽑아내보자
    print((word, r))  #8칸 띄고, 소수점 4자리

#한글폰트 지원이 되지 않기 때문에 별도로 이걸 깔아줘야 함
import matplotlib as mpl
import matplotlib.pyplot as plt
 
%config InlineBackend.figure_format = 'retina'
 
!apt -qq -y install fonts-nanum
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
#mpl.font_manager._rebuild()
#[출처] [Google Colab] 구글 코랩 한글 적용 문제 대응, Matplotlib|작성자 넬티아

from wordcloud import WordCloud
from wordcloud import ImageColorGenerator

wc = WordCloud(font_path=fontpath, width=1000, height=1000, scale=3.0, max_font_size=250, background_color = 'white')
full_responsibilities = wc.generate_from_frequencies(passwords)
plt.figure(figsize=(10,10))
plt.imshow(full_responsibilities)
plt.imshow(full_responsibilities)