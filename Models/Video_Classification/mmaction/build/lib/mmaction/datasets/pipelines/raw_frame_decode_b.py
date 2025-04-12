from mmaction.datasets.pipelines import PIPELINES
import numpy as np

@PIPELINES.register_module()
class RawFrameDecode_B:
    """커스텀 RawFrameDecode_B 데이터 로더"""

    def __init__(self, io_backend='disk'):
        self.io_backend = io_backend

    def __call__(self, results):
        """
        데이터를 불러오고 변환하는 메서드
        :param results: MMACTION2 데이터셋이 전달하는 입력 딕셔너리
        :return: 변환된 데이터
        """
        frame_dir = results['frame_dir']  # 프레임이 저장된 디렉토리
        total_frames = results['total_frames']  # 전체 프레임 개수

        # 예제: 모든 프레임을 numpy 배열로 변환
        frame_list = []
        for i in range(1, total_frames + 1):
            frame_path = f"{frame_dir}/img_{i:05d}.jpg"  # 파일 경로 예제
            frame = self.load_image(frame_path)
            frame_list.append(frame)

        results['imgs'] = np.array(frame_list)
        results['modality'] = 'RGB'
        results['frame_num'] = total_frames
        return results

    def load_image(self, path):
        """이미지를 로드하는 헬퍼 함수"""
        import cv2
        from PIL import Image
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV는 BGR이므로 RGB로 변환
        return img

    def __repr__(self):
        return f"{self.__class__.__name__}(io_backend={self.io_backend})"
