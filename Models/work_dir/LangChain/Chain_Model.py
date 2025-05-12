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


# 한글 폰트 등록
# Windows에 설치된 기본 한글 폰트 사용
pdfmetrics.registerFont(TTFont('Malgun', 'C:/Windows/Fonts/malgun.ttf'))
pdfmetrics.registerFont(TTFont('MalgunBold', 'C:/Windows/Fonts/malgunbd.ttf'))

load_dotenv()

# API KEY 정보로드
API_KEY = os.getenv("OPEN_API_KEY")

video_name = "test7"

work_dir = "C:/Users/Noh/github/Accident_Prediction_Prevent/Models/work_dir/"

pdf_path = work_dir + "LangChain/pdf_data/231107_과실비율인정기준_온라인용.pdf"
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

# 단계 1 : 문서 로드
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 단계 2 : 문서 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
)
split_documents = text_splitter.split_documents(docs)

# 단계 3 : 임베딩
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"  # 더 가벼운 모델
)

# 단계 4 : 벡터 스토어 생성
vectorstore = FAISS.from_documents(split_documents, embeddings)

# 단계 5 : 검색기 생성
retriever = vectorstore.as_retriever()

# 단계 6 : 프롬프트 생성
prompt = PromptTemplate.from_template(
    """
    당신은 교통사고 분석 전문가입니다.
    사용자 질문과 문서 내용을 바탕으로 다음과 같은 구조로 사고 정보를 정리해주세요.

    [사건정보]
    사건분류: (예: 차대 보행자 / 차대 이륜차 / 차대 자전거 / 차대차)
    사고장소: (도로 형태 또는 사고 발생 지점 요약)
    객체 A 상태: (행동/위치/신호 상태 등)
    객체 B 상태: (행동/위치/신호 상태 등)
    사고유형번호: (예: 보9, 차15-1 등)

    사고 상황: (구체적으로 서술해주세요. 예: 신호등이 있는 교차로에서 보행자가 신호를 무시하고 횡단보도를 건너는 상황)

    [기본 과실비율]
    사고유형에 따라 적용되는 기본 과실비율을 정리해주세요.

    [과실비율 조정 요소]
    사고 발생에 따라 적용 가능한 가감 요소를 아래와 같이 정리해주세요.
    항목과 수치를 문서에서 찾을 수 있을 경우 구체적으로 적어주세요.
    예: 야간 시야장애: +10%, 어린이 보호구역: -15% 등

    [관련 법률]
    문서에 명시된 관련 법령이 있다면 조문 번호와 함께 정리해주세요.
    예: 도로교통법 제5조 (신호 또는 지시에 따를 의무)

    [참고 판례]
    해당 사고유형과 관련된 판례가 있는 경우 아래 형식으로 제시해주세요.
    - 법원명 / 선고일 / 사건번호
    - 핵심 내용 요약
    - 과실비율 요약 (있다면)

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

question = f"""사고 장소 {one} 사고 유형 {two} 사고 객체 A의 진행 방향 {three} 사고 객체 B의 진행 방향 {four}에 해당하는 사고를 찾아줘. """

# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변과 관련 이미지를 출력합니다.
question = question
response = chain.invoke(question)

### 보고서 생성

def create_report_pdf(question, response, pdf_image=None, output_filename=None):
    # 출력 디렉토리
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # 기존 스타일 수정 (한글 폰트 적용)
    styles['Title'].fontName = 'MalgunBold'
    styles['Title'].alignment = 1  # 가운데 정렬
    styles['Title'].fontSize = 16
    styles['Title'].spaceAfter = 12
    
    # Normal 스타일도 한글 폰트로 수정
    styles['Normal'].fontName = 'Malgun'
    styles['Normal'].fontSize = 11
    
    # 새 스타일 추가 (한글 폰트 적용)
    styles.add(ParagraphStyle(name='MyHeading',
                              fontName='MalgunBold',
                              fontSize=14,
                              spaceAfter=6))
    
    styles.add(ParagraphStyle(name='MyNormal',
                              fontName='Malgun',
                              fontSize=11,
                              leading=14))  # 줄 간격
    
    # 문서 내용
    elements = []
    
    # 제목
    title = Paragraph("사고 분석 보고서", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 0.25*inch))
    
    # 날짜 추가
    date_str = datetime.datetime.now().strftime("%Y년 %m월 %d일")
    date_paragraph = Paragraph(f"작성일: {date_str}", styles['Normal'])
    elements.append(date_paragraph)
    elements.append(Spacer(1, 0.25*inch))

    if pdf_image:
        # 이미지 추가
        # PIL 이미지를 ReportLab 이미지로 변환
        img_buffer = BytesIO()
        pdf_image.save(img_buffer, format='PNG')
        img_data = img_buffer.getvalue()
        img_buffer.close()
        
        # 페이지에 맞는 적절한 이미지 크기 계산
        # 여유 공간 계산 (제목, 날짜, 답변 등이 차지하는 공간 고려)
        text_content_height = 4.5 * inch  # 대략적인 텍스트 내용 높이 예상값
        available_height = letter[1] - 2*72 - text_content_height  # 페이지 높이 - 상하 여백 - 텍스트 내용
        
        # 이미지 크기 조정 (너비와 높이 제한)
        max_width = 5 * inch  # 페이지 너비를 고려한 최대 이미지 너비 (기존 6인치에서 줄임)
        max_height = available_height - 0.5 * inch  # 약간의 추가 여유 공간
        
        # 원본 비율 유지하면서 크기 제한
        img_width = min(pdf_image.width, max_width)
        img_height = (pdf_image.height * img_width) / pdf_image.width
        
        # 높이가 제한을 초과하면 높이 기준으로 다시 계산
        if img_height > max_height:
            img_height = max_height
            img_width = (pdf_image.width * img_height) / pdf_image.height
        
        # 이미지 추가
        img_obj = ReportImage(BytesIO(img_data), width=img_width, height=img_height)
        elements.append(img_obj)
        elements.append(Spacer(1, 0.15*inch))
    
    # # 질문 섹션
    # elements.append(Paragraph("질문:", styles['MyHeading']))
    # # 줄바꿈을 HTML <br/>로 대체
    # formatted_question = question.replace('\n', '<br/>')
    # elements.append(Paragraph(formatted_question, styles['MyNormal']))
    # elements.append(Spacer(1, 0.25*inch))
    
    # 답변 섹션
    # 줄바꿈을 HTML <br/>로 대체
    formatted_response = response.replace('\n', '<br/>')
    elements.append(Paragraph(formatted_response, styles['MyNormal']))
    elements.append(Spacer(1, 0.25*inch))
    
    # PDF 생성
    doc.build(elements)
    print(f"✅ 보고서가 생성되었습니다: {output_path}")
    
    return output_path

# 이미지 가져오기
def get_relevant_images(pdf_path, response):
    """LangChain 응답과 관련된 이미지들을 추출합니다."""
    relevant_docs = vectorstore.similarity_search(response, k=1)

    page_num = relevant_docs[0].metadata.get('page', 0)  # 첫 번째 문서의 페이지 번호를 사용
    
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    image_list = page.get_images(full=True)
        
    if not image_list:  # 이미지가 있는 경우에만
        return None
    
    xref = image_list[0][0]  # 첫 번째 이미지 사용
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    image = Image.open(io.BytesIO(image_bytes))
    
    if image:
        print(f"✅ 관련 이미지가 추출되었습니다: {image_list[0]}")

    return image

# 보고서 생성
def generate_accident_report(pdf_path, response):
    pdf_path = pdf_path
    response = response
    # 이미지 가져오기
    pdf_image = get_relevant_images(pdf_path, response)
    
    # 타이틀에 사고 정보 포함
    report_title = f"사고분석_{one}_{two}.pdf"
    
    # PDF 보고서 생성
    pdf_path = create_report_pdf(
        question=question,
        response=response,
        pdf_image=pdf_image,
        output_filename=report_title
    )
    
    return pdf_path

# 보고서 생성 실행
report_file = generate_accident_report(pdf_path, response)