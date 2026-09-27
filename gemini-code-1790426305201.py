import os
import time
import datetime
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from apscheduler.schedulers.background import BackgroundScheduler

# ==========================================
# 1. API 토큰 및 설정값 (발급받은 키 입력)
# ==========================================
TOKENS = {
    "instagram": os.getenv("INSTAGRAM_ACCESS_TOKEN", "YOUR_INSTAGRAM_TOKEN"),
    "youtube": os.getenv("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY"),
    "twitter": os.getenv("TWITTER_BEARER_TOKEN", "YOUR_TWITTER_BEARER_TOKEN"),
}

# 해시태그 그룹 정의 (0: 기본/브랜딩, 1: 바이럴/트렌드, 2: 틈새/키워드)
HASHTAG_GROUPS = {
    0: ["#일상", "#소통", "#데일리"],
    1: ["#FYP", "#바이럴", "#인기게시물", "#쇼츠"],
    2: ["#개발자", "#강화학습", "#파이썬", "#AI"],
}

# ==========================================
# 2. 멀티 플랫폼 데이터 수집 모듈 (Reward Extractor)
# ==========================================
class SocialMediaCollector:
    """플랫폼별 정식 API를 통해 2시간 후 성과 수치를 수집하고 Reward를 계산하는 클래스"""

    @staticmethod
    def get_instagram_metrics(media_id: str) -> float:
        """인스타그램 인사이트 수집 (도달, 저장, 좋아요)"""
        url = f"https://graph.facebook.com/v18.0/{media_id}/insights"
        params = {
            "metric": "reach,saved,likes",
            "access_token": TOKENS["instagram"],
        }
        try:
            res = requests.get(url, params=params, timeout=10).json()
            metrics = {
                item["name"]: item["values"][0]["value"]
                for item in res.get("data", [])
            }
            reach = metrics.get("reach", 0)
            saved = metrics.get("saved", 0)
            likes = metrics.get("likes", 0)

            # 저장 수(5점)와 도달 수(1점)에 가중치 부여
            reward = (reach * 1.0) + (saved * 5.0) + (likes * 2.0)
            print(f"📸 [Instagram Metric] Reach: {reach}, Saved: {saved}, Reward: {reward}")
            return reward
        except Exception as e:
            print(f"❌ Instagram 수집 실패 (기본값 처리): {e}")
            return 10.0  # 예외 발생 시 기본 보상 처리

    @staticmethod
    def get_youtube_metrics(video_id: str) -> float:
        """유튜브 쇼츠 통계 수집 (조회수, 좋아요)"""
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "statistics",
            "id": video_id,
            "key": TOKENS["youtube"],
        }
        try:
            res = requests.get(url, params=params, timeout=10).json()
            stats = res["items"][0]["statistics"]
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))

            reward = (views * 1.0) + (likes * 3.0)
            print(f"🎬 [YouTube Metric] Views: {views}, Likes: {likes}, Reward: {reward}")
            return reward
        except Exception as e:
            print(f"❌ YouTube 수집 실패 (기본값 처리): {e}")
            return 10.0

    @staticmethod
    def get_twitter_metrics(tweet_id: str) -> float:
        """트위터(X) 트윗 metrics 수집 (리트윗, 좋아요, 답글)"""
        url = f"https://api.twitter.com/2/tweets/{tweet_id}?tweet.fields=public_metrics"
        headers = {"Authorization": f"Bearer {TOKENS['twitter']}"}
        try:
            res = requests.get(url, headers=headers, timeout=10).json()
            metrics = res["data"]["public_metrics"]
            retweets = metrics.get("retweet_count", 0)
            likes = metrics.get("like_count", 0)

            reward = (retweets * 5.0) + (likes * 1.0)
            print(f"🐦 [Twitter Metric] Retweets: {retweets}, Likes: {likes}, Reward: {reward}")
            return reward
        except Exception as e:
            print(f"❌ Twitter 수집 실패 (기본값 처리): {e}")
            return 10.0

    @classmethod
    def fetch_reward(cls, platform: str, post_id: str) -> float:
        if platform == "instagram":
            return cls.get_instagram_metrics(post_id)
        elif platform == "youtube":
            return cls.get_youtube_metrics(post_id)
        elif platform == "twitter":
            return cls.get_twitter_metrics(post_id)
        else:
            raise ValueError("지원하지 않는 플랫폼입니다.")


# ==========================================
# 3. 강화학습 커스텀 환경 (Gymnasium Environment)
# ==========================================
class SNSOptimizerEnv(gym.Env):
    """업로드 요일과 상태를 받아 최적의 시각(0~23시)과 태그(0~2)를 결정하는 RL 환경"""

    def __init__(self):
        super(SNSOptimizerEnv, self).__init__()

        # [State]: 0: 요일(0~6, 월~일), 1: 최근 평균 보상 점수(0~500)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([6, 500]), dtype=np.float32
        )

        # [Action]: 0~71 (24개 시간대 x 3개 해시태그 그룹)
        # Action 0 = 0시 + 태그0, Action 1 = 0시 + 태그1 ... Action 71 = 23시 + 태그2
        self.action_space = spaces.Discrete(72)

        self.current_day = 0
        self.last_reward = 50.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = datetime.datetime.now().weekday()
        state = np.array([float(self.current_day), float(self.last_reward)], dtype=np.float32)
        return state, {}

    def step(self, action, external_reward=None):
        hour = action // 3
        tag_group = action % 3

        # 외부에서 실측 보상(API 수집값)이 들어오면 해당 값 사용, 없으면 가상 알고리즘 보상 시뮬레이션
        if external_reward is not None:
            reward = external_reward
        else:
            # 기본 가상 환경 반응 함수 (금/토 저녁 19~21시 + 바이럴 태그일 때 피크)
            reward = 10.0
            if self.current_day in [4, 5] and hour in [19, 20, 21]:
                reward += 100.0
                if tag_group == 1:
                    reward += 50.0
            reward = max(0.0, reward + np.random.normal(0, 5))

        self.last_reward = reward
        self.current_day = (self.current_day + 1) % 7
        next_state = np.array([float(self.current_day), float(self.last_reward)], dtype=np.float32)

        return next_state, reward, False, False, {}


# ==========================================
# 4. 전체 파이프라인 관리자 (Pipeline Manager)
# ==========================================
class SNSOptimizationPipeline:
    def __init__(self):
        self.env = SNSOptimizerEnv()
        self.model = PPO("MlpPolicy", self.env, verbose=0)
        self.scheduler = BackgroundScheduler()
        self.scheduler.start()

    def train_initial_model(self, timesteps=10000):
        """기본 콜드스타트 방지용 사전 학습"""
        print("🧠 [RL] 사전 모델 학습 시작...")
        self.model.learn(total_timesteps=timesteps)
        print("✅ [RL] 사전 모델 학습 완료")

    def predict_best_schedule(self):
        """오늘 기준 최적의 업로드 시각 및 태그 그룹 예측"""
        current_day = datetime.datetime.now().weekday()
        state = np.array([float(current_day), 50.0], dtype=np.float32)

        action, _ = self.model.predict(state, deterministic=True)
        best_hour = int(action // 3)
        best_tag_group = int(action % 3)

        print(f"\n🎯 [AI 추천] 오늘 추천 포스팅 시간: {best_hour}시")
        print(f"🏷️ [AI 추천] 추천 해시태그 그룹: {HASHTAG_GROUPS[best_tag_group]}")

        return best_hour, best_tag_group, action

    def delayed_feedback_job(self, platform: str, post_id: str, action: int):
        """2시간 뒤 실행되어 API 수치를 긁어오고 RL 모델을 재학습시키는 작업"""
        print(f"\n⏰ [2시간 경과] {platform} (Post ID: {post_id}) 성과 수집 시작...")

        # 1. API 호출로 실측 보상 수집
        real_reward = SocialMediaCollector.fetch_reward(platform, post_id)

        # 2. 환경 업데이트 및 모델 추가 학습 (Fine-Tuning)
        self.env.step(action, external_reward=real_reward)
        self.model.learn(total_timesteps=100)
        print(f"🔄 [RL] 실측 보상({real_reward}) 반영 및 AI 모델 가중치 업데이트 완료")

    def register_post(self, platform: str, post_id: str, action: int):
        """게시글 업로드 완료 후 2시간 뒤 피드백 스케줄링 등록"""
        run_time = datetime.datetime.now() + datetime.timedelta(hours=2)
        # 테스트 목적으로는 아래 timedelta를 seconds=10 등으로 변경 가능합니다.

        self.scheduler.add_job(
            self.delayed_feedback_job,
            "date",
            run_date=run_time,
            args=[platform, post_id, action],
        )
        print(f"📅 [스케줄러 등록] {run_time.strftime('%H:%M:%S')}에 2시간 후 보상 수집 작업이 실행됩니다.")


# ==========================================
# 5. 실행부 (Execution Main)
# ==========================================
if __name__ == "__main__":
    print("🚀 SNS RL 최적화 파이프라인 시스템을 시작합니다.")

    # 1. 파이프라인 인스턴스 생성
    pipeline = SNSOptimizationPipeline()

    # 2. RL 모델 오프라인 시뮬레이션 미리 학습
    pipeline.train_initial_model(timesteps=5000)

    # 3. 오늘 올려야 할 최적 시각/태그 예측
    best_hour, best_tag_group, selected_action = pipeline.predict_best_schedule()

    # 4. (가정) 인스타그램에 실제로 글을 올리고 Media ID(1798123456)를 받아왔다고 가정
    simulated_post_id = "1798123456"
    target_platform = "instagram"

    # 5. 글을 올렸으므로 2시간 뒤 자동 수집 & RL 학습 스케줄러 등록
    pipeline.register_post(
        platform=target_platform, post_id=simulated_post_id, action=selected_action
    )

    # 파이썬 프로세스가 대기하도록 설정 (실제 서버에서는 FastAPI / Celery 환경에서 백그라운드로 구동)
    print("\n⏳ 백그라운드 스케줄러 대기 중... (Ctrl+C 로 종료)")
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pipeline.scheduler.shutdown()
        print("👋 파이프라인 프로세스를 종료합니다.")