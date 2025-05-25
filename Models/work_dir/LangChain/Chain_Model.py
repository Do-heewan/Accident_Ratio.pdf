# API KEY를 환경변수로 관리하기 위한 설정 파일
import os 
import json
import datetime
import fitz  # PyMuPDF
from PIL import Image
import io
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as ReportImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak


# 한글 폰트 등록
# Windows에 설치된 기본 한글 폰트 사용
pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold', 'C:/Windows/Fonts/malgunbd.ttf'))

load_dotenv()

# API KEY 정보로드
API_KEY = os.getenv("OPEN_API_KEY")

video_name = "bb_1_220122_vehicle_229_34825"

work_dir = "C:/Users/Noh/github/Accident_Prediction_Prevent/Models/work_dir/"

accident_ratio_pdf_path = work_dir + "LangChain/pdf_data/231107_과실비율인정기준_온라인용.pdf"
traffic_law_pdf_path = work_dir + "LangChain/pdf_data/도로교통법.pdf"

json_path = work_dir + "datasets/results/" + video_name + "_classification.json"

output_dir = work_dir + "/datasets/results/"

### 사고 정보 매핑 파일 로드
def load_mapping_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
    
json_file = load_mapping_file(json_path) # 모델 아웃풋 파일

one = json_file[0]["accident_place"]
two = json_file[0]["accident_place_feature"]
three = json_file[0]["object_A"]
four = json_file[0]["object_B"]

### 랭체인 실행
print("랭체인을 실행합니다...")
# 단계 1 : 문서 로드
loader = PyPDFLoader(accident_ratio_pdf_path)
docs = loader.load()

traffic_loader = PyPDFLoader(traffic_law_pdf_path)
traffic_docs = traffic_loader.load()

# 단계 2 : 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
)
split_documents = text_splitter.split_documents(docs)
split_traffic_documents = text_splitter.split_documents(traffic_docs)

# 단계 3 : 임베딩
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 더 가벼운 모델
)

# 단계 4 : 벡터 스토어 생성
vectorstore = FAISS.from_documents(split_documents, embeddings)
traffic_vectorstore = FAISS.from_documents(split_traffic_documents, embeddings)

# 단계 5 : 검색기 생성
retriever = vectorstore.as_retriever()
traffic_retriever = traffic_vectorstore.as_retriever()

# 단계 6 : 프롬프트 생성
prompt = PromptTemplate.from_template(
    """
    당신은 교통사고 분석 전문가입니다.
    사용자 질문과 문서 내용을 바탕으로 다음과 같은 구조로 사고 정보를 정리해주세요.
    ~~ 입니다. ~~ 답변 드리겠습니다. 등의 답변은 하지 마세요.

    사고유형번호에 대해 물어보면 예시와 동일한 유형의 사고유형 번호를 알려주세요. 없으면 답하지 마세요. (예시 : 차15-1, 보9 등)

    사건정보에 대해 물어보면 다음과 같은 형식으로 답변해주세요. :
    [사건분류: (예: 차대 보행자 / 차대 이륜차 / 차대 자전거 / 차대차)
    사고장소: (도로 형태 또는 사고 발생 지점 요약)
    객체 A 상태: (행동/위치/신호 상태 등)
    객체 B 상태: (행동/위치/신호 상태 등)

    사고 상황: (구체적으로 서술해주세요. 예: 신호등이 있는 교차로에서 보행자가 신호를 무시하고 횡단보도를 건너는 상황)]

    기본 과실비율에 대해 물어보면 다음과 같은 형식으로 답변해주세요. :
    [기본 과실비율 : (예: A50:B50 / A70:B30)]

    과실 비율 조정 요소에 대해 물어보면 다음과 같은 형식으로 답변해주세요. :
    사고 발생에 따라 적용 가능한 가감 요소를 정리해주고, 문서에서 찾을 수 있는 항목과 수치만을 구체적으로 적어주세요.

    관련 법률에 대해 물어보면 다음과 같은 형식으로 답변해주세요. :
    [관련 법률 : (예: 도로교통법 제5조 (신호 또는 지시에 따를 의무))]

    참고 판례에 대해 물어보면 다음과 같은 형식으로 답변해주세요. :
    - 법원명 : 
    - 선고일 :
    - 사건번호
    - 핵심 내용 요약
    - 과실비율 요약 (있다면)

    사고 요약에 대해 물어보면 다음과 같은 형식으로 답변해주세요. :
    [사고 요약 : (사고 유형, 장소, 객체 A와 B의 상태)]

    질문:
    {question}

    참고 문서:
    {context}

    위 형식을 정확히 따르고, 한국어로 정리해주세요.
    """
)

# 단계 7: 언어모델(LLM) 생성
# 모델(LLM) 을 생성합니다.
llm = ChatOpenAI(model_name="gpt-4o",
                openai_api_key=API_KEY,
                temperature=0)

# 단계 8: 체인(Chain) 생성
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

accident = f"사고 장소 {one} 사고 유형 {two} 사고 객체 A의 진행 방향 {three} 사고 객체 B의 진행 방향 {four}에 해당하는 사고"

questions = [accident + "의 사고유형번호는?",
            accident + "의 사건정보는?",
            accident + "의 기본 과실 비율은?",
            accident + "의 과실 비율 조정 요소는?",
            accident + "의 관련 법률은?",
            accident + "의 참고 판례는?",
            accident + "요약 정리"
            ]

res = []
for i, question in enumerate(questions):
    # 질문에 변수를 삽입합니다.
    question = question.format(one=one, two=two, three=three, four=four)
    
    # 질문을 체인에 전달하여 답변을 생성합니다.
    print(f"{i+1}번째 질문에 대한 답변 생성 중입니다...")
    response = chain.invoke(question)
    res.append(response)

print("✅답변 생성 완료")


print("쟁점 분석을 시작합니다...")
summary = res[6]
issue_prompt = PromptTemplate.from_template(
    '''
    다음은 교통사고 분석 보고서입니다.
    사용자 질문과 문서 내용을 바탕으로 다음과 같은 구조로 사고 정보를 정리해주세요.
    ~~ 입니다. ~~ 답변 드리겠습니다. 등의 답변은 하지 마세요.
    
    주요 쟁점에 대해 질문하면 다음과 같이 답변해주세요.
    [주요 쟁점 : 사고에서 핵심적으로 판단해야 할 포인트와 과실비율 산정에 영향을 줄 수 있는 주요 쟁점을 서술해 주세요.]
    
    결정 근거에 대해 질문하면 다음과 같이 답변해주세요.
    [결정 근거 : 사고의 판단에 영향을 미친 요소들과 판단 근거, 판단의 주요 논리를 서술해 주세요.]
    
    법률 키워드에 대해 질문하면 다음과 같이 답변해주세요.
    [법률 키워드 : 사고의 법률적 해석 및 판단에 중요한 핵심 키워드를 3~5개 추출해 주세요. 예: 선진입 우선, 진로변경, 교차로 통행 우선, 신호위반, 우측차 우선 등]

    분석 요약:
    {analysis}
    '''
)

issue_chain = (
    {"analysis": RunnablePassthrough()}
    | issue_prompt
    | llm
    | StrOutputParser()
)

questions = [summary + "의 주요 쟁점은 무엇인가요?",
            summary + "의 결정 근거는 무엇인가요?",
            summary + "의 법률 키워드는 무엇인가요?"]

summary_res = []
for i, question in enumerate(questions):
    # 질문에 변수를 삽입합니다.
    question = question.format(summary=summary)
    
    # 질문을 체인에 전달하여 답변을 생성합니다.
    print(f"{i+1}번째 질문에 대한 답변 생성 중입니다...")
    response = issue_chain.invoke(question)
    summary_res.append(response)

print("✅쟁점 답변 생성 완료")

print("관련 법률 조항을 검색합니다...")
law_prompt = PromptTemplate.from_template(
    '''
    다음은 교통사고 분석 결과입니다. 관련 도로교통법 조항을 아래 형식으로 2~3개 선정하여 설명하세요.

    사고:
    {question}

    [참고 문서]
    --------------------
    {law_context}
    --------------------

    🔹 각 조항은 아래 형식으로 작성해 주세요:
    제13조 제1항:
    "모든 차의 운전자는 ..."
    ➞  ... 상황에 적용되며, 이 사건에서는 ... 이유로 판단됨.

    ✳️ 유사 적용 가능한 조항이 있다면 추가로 간단히 설명해 주세요.
    '''
)

related_law_chain = (
    {"law_context": traffic_retriever, "question": RunnablePassthrough()}
    | law_prompt
    | llm
    | StrOutputParser()
)

traffic_response = related_law_chain.invoke(summary)
print("✅관련 법률 조항 검색 완료")

### 보고서 생성
def create_report_pdf(response, summary, traffic_response, output_filename=None):
    # 출력 디렉토리
    os.makedirs(output_dir, exist_ok=True)

    res = response
    summary_res = summary
    traffic_response = traffic_response
    
    # 파일명 생성 (지정되지 않은 경우 타임스탬프 사용)
    if not output_filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"accident_report_{timestamp}.pdf"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # PDF 문서 설정
    doc = SimpleDocTemplate(output_path, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # 스타일 설정 - 기존 스타일 가져오기
    styles = getSampleStyleSheet()
    
    # 대제목 스타일 수정 (한글 폰트 적용)
    styles['Title'].fontName = 'MalgunBold'
    styles['Title'].alignment = 1  # 가운데 정렬
    styles['Title'].fontSize = 16
    styles['Title'].spaceAfter = 12

    # 중제목
    styles['Heading1'].fontName = 'MalgunBold'
    styles['Heading1'].fontSize = 14
    
    # Normal 스타일도 한글 폰트로 수정
    styles['Normal'].fontName = 'Malgun'
    styles['Normal'].fontSize = 11

    # 새 스타일 추가 (한글 폰트 적용)
    styles.add(ParagraphStyle(name='DateStyle',
                              fontName='MalgunBold',
                              fontSize=11,
                              alignment=2))  # 오른쪽 정렬
    
    styles.add(ParagraphStyle(name='img_name',
                              fontName='Malgun',
                              fontSize=9,
                              alignment=1))  # 가운데 정렬
    
    styles.add(ParagraphStyle(name='acc_ratio',
                              fontName='MalgunBold',
                              fontSize=20,
                              alignment=1))  # 가운데 정렬
    
    # 문서 내용
    elements = []
    
    # 제목
    title = Paragraph("사고 분석 보고서", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.25*inch))
    
    # 날짜 추가
    date_str = datetime.datetime.now().strftime("%Y년 %m월 %d일")
    date_paragraph = Paragraph(f"작성일: {date_str}", styles['DateStyle'])
    elements.append(date_paragraph)
    elements.append(Spacer(1, 0.25*inch))

    # 기본 정보 입력란
    basic_info_title = Paragraph("[사고 발생 정보]", styles['Heading1'])
    elements.append(basic_info_title)
    basic_info = Paragraph("사고 발생일 : <br/> 사고 지점 : ", styles['Normal'])
    elements.append(basic_info)
    elements.append(Spacer(1, 0.5*inch))
    
    # 답변 섹션
    acc_num = res[0].replace("사고유형번호: ", "")
    acc_info = res[1].replace("[", "").replace("]", "")
    basic_ratio = res[2][11:18]
    adjust_ratio = res[3].replace("과실 비율 조정 요소:\n\n", "").replace("]", "")
    related_law = res[4].replace("[관련 법률 : ", "").replace("]", "")
    reference_case = res[5].replace("[", "").replace("]", "")
    main_issue = summary_res[0].replace("[주요 쟁점 : ", "").replace("]", "")
    decision_basis = summary_res[1].replace("[결정 근거 : ", "").replace("]", "")
    law_keywords = summary_res[2].replace("[법률 키워드 : ", "").replace("]", "").split(", ")
    traffic_response = traffic_response

    # 줄바꿈을 HTML <br/>로 대체
    acc_num_p = Paragraph("<" + acc_num + ">", styles['img_name'])
    acc_info = Paragraph(acc_info.replace('\n', '<br/>'), styles['Normal'])
    basic_ratio = Paragraph(basic_ratio, styles['acc_ratio'])
    adjust_ratio = Paragraph(adjust_ratio.replace('\n', '<br/>'), styles['Normal'])
    related_law = Paragraph(related_law, styles['Normal'])
    reference_case = Paragraph(reference_case.replace('\n', '<br/>'), styles['Normal'])
    main_issue = Paragraph(main_issue, styles['Normal'])
    decision_basis = Paragraph(decision_basis, styles['Normal'])

    traffic_response = Paragraph(traffic_response.replace('\n', '<br/>'), styles['Normal'])

    # AI 분석 사고 정보 및 상황
    acc_info_title = Paragraph("[AI 분석 사고 정보 및 상황]", styles['Heading1'])
    elements.append(acc_info_title)
    elements.append(acc_info)
    elements.append(Spacer(1, 0.25*inch))

    # 사고유형에 맞는 이미지 첨부
    img_dir = os.path.join(work_dir, "LangChain/pdf_images/")
    image = os.path.join(img_dir, f"{acc_num}.jpeg")
    if os.path.exists(image):
        img = Image.open(image)
        img = img.convert("RGB")
        img_stream = BytesIO()
        img.save(img_stream, format="JPEG")
        img_stream.seek(0)
        
        # 이미지 추가
        report_image = ReportImage(img_stream, width=6*inch, height=4*inch)
        # report_image = ReportImage(img_stream, width=doc.width, height=doc.width * 0.75)
        elements.append(report_image)
        elements.append(Spacer(1, 0.25*inch))
    else:
        print(f"❌ 이미지 파일이 존재하지 않습니다: {image}")

    elements.append(acc_num_p)
    elements.append(Spacer(1, 0.25*inch))

    # ✅ 다음 페이지로 강제 이동
    elements.append(PageBreak())    

    # AI 분석 기본 과실 비율
    basic_ratio_title = Paragraph("[AI 분석 기본 과실 비율]", styles['Heading1'])
    elements.append(basic_ratio_title)
    elements.append(basic_ratio)
    elements.append(Spacer(1, 0.5*inch))

    # 과실비율 조정 요소
    adjust_ratio_title = Paragraph("[AI 분석 과실비율 조정 요소]", styles['Heading1'])
    elements.append(adjust_ratio_title)
    elements.append(adjust_ratio)
    elements.append(Spacer(1, 0.5*inch))

    # 관련 법률
    related_law_title = Paragraph("[AI 분석 관련 법률]", styles['Heading1'])
    elements.append(related_law_title)
    elements.append(related_law)
    elements.append(Spacer(1, 0.5*inch))

    # 참고 판례
    reference_case_title = Paragraph("[AI 분석 참고 판례]", styles['Heading1'])
    elements.append(reference_case_title)
    elements.append(reference_case)
    elements.append(Spacer(1, 0.5*inch))

    # 주요 쟁점
    main_issue_title = Paragraph("[AI 분석 주요 쟁점]", styles['Heading1'])
    elements.append(main_issue_title)
    elements.append(main_issue)
    elements.append(Spacer(1, 0.5*inch))

    # 결정 근거
    decision_basis_title = Paragraph("[AI 분석 결정 근거]", styles['Heading1'])
    elements.append(decision_basis_title)
    elements.append(decision_basis)
    elements.append(Spacer(1, 0.5*inch))

    # 법률 키워드

    # 도로교통법
    traffic_response_title = Paragraph("[도로교통법 관련 조항]", styles['Heading1'])
    elements.append(traffic_response_title)
    elements.append(traffic_response)
    elements.append(Spacer(1, 0.25*inch))
    
    # PDF 생성
    doc.build(elements)
    print(f"✅ 보고서가 생성되었습니다: {output_path}")
    
    return output_path

# 보고서 생성
def generate_accident_report(response, summary_res, traffic_response):
    response = response
    summary_res = summary_res
    traffic_response = traffic_response

    report_title = f"{video_name}.pdf"
    
    # PDF 보고서 생성
    pdf_path = create_report_pdf(
        response=response,
        summary=summary_res,
        traffic_response=traffic_response,
        output_filename=report_title
    )
    
    return pdf_path

# 보고서 생성 실행
report_file = generate_accident_report(res, summary_res, traffic_response)