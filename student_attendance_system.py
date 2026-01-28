# recognize_and_attendance.
import cv2
import pickle
import csv
from datetime import datetime
import os

class StudentAttendanceSystem:
    def __init__(self, model_path="face_model.yml", labels_path="labels.pkl", 
                 confidence_threshold=100, attendance_file="attendance.csv"):
        """
        学生出席認識システム
        
        Args:
            model_path: 訓練済みモデルのパス
            labels_path: ラベルマップのパス
            confidence_threshold: 認識の信頼度閾値（低いほど厳格）
            attendance_file: 出席記録ファイル
        """
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.attendance_file = attendance_file
        
        # 今日の出席セット
        self.today_attendance = set()
        
        # 認識バッファ（安定した認識のため）
        self.recognition_buffer = {}
        self.buffer_threshold = 3  # 連続認識回数
        
        self.setup_system()
    
    def setup_system(self):
        """システムの初期化"""
        try:
            # モデルとラベルの読み込み
            self.load_model_and_labels()
            
            # カメラと顔検出器の初期化
            self.setup_camera()
            
            # 今日の出席状況を読み込み
            self.load_today_attendance()
            
            # 出席ファイルの準備
            self.setup_attendance_file()
            
            print("✅ システム初期化完了")
            
        except Exception as e:
            print(f"❌ システム初期化エラー: {e}")
            raise
    
    def load_model_and_labels(self):
        """モデルとラベルマップの読み込み"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
        
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"ラベルファイルが見つかりません: {self.labels_path}")
        
        # モデル読み込み
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.read(self.model_path)
        
        # ラベルマップ読み込み
        with open(self.labels_path, "rb") as f:
            self.label_map = pickle.load(f)
        
        print(f"📚 {len(self.label_map)}人の学生データを読み込みました")
    
    def setup_camera(self):
        """カメラと顔検出器の初期化"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("カメラを開けません")
        
        # カメラ設定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # 顔検出器
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        
        if self.detector.empty():
            raise RuntimeError("顔検出器の読み込みに失敗しました")
    
    def setup_attendance_file(self):
        """出席ファイルの準備"""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["日付", "時刻", "名前", "学籍番号", "信頼度"])
    
    def load_today_attendance(self):
        """今日の出席状況を読み込み"""
        today = datetime.now().strftime("%Y-%m-%d")
        self.today_attendance = set()
        
        if os.path.exists(self.attendance_file):
            try:
                with open(self.attendance_file, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # ヘッダーをスキップ
                    for row in reader:
                        if len(row) >= 4 and row[0] == today:
                            name_id = f"{row[2]}_{row[3]}"
                            self.today_attendance.add(name_id)
                
                print(f"📋 本日の出席者: {len(self.today_attendance)}人")
                
            except Exception as e:
                print(f"警告: 既存の出席データの読み込みに失敗: {e}")
    
    def parse_name_id(self, name_id):
        """name_studentID形式から名前と学籍番号を分離"""
        try:
            if "_" in name_id:
                parts = name_id.rsplit("_", 1)  # 最後の_で分割
                return parts[0], parts[1]
            else:
                return name_id, "000"
        except:
            return "不明", "000"
    
    def record_attendance(self, name_id, confidence):
        """出席を記録"""
        if name_id in self.today_attendance:
            return False  # 既に記録済み
        
        try:
            name, student_id = self.parse_name_id(name_id)
            current_time = datetime.now()
            date_str = current_time.strftime("%Y-%m-%d")
            time_str = current_time.strftime("%H:%M:%S")
            
            # CSVに記録
            with open(self.attendance_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([date_str, time_str, name, student_id, f"{confidence:.1f}"])
            
            # 今日の出席セットに追加
            self.today_attendance.add(name_id)
            
            print(f"✅ 出席記録: {name}（学籍番号: {student_id}）- 信頼度: {confidence:.1f}")
            return True
            
        except Exception as e:
            print(f"❌ 出席記録エラー: {e}")
            return False
    
    def update_recognition_buffer(self, face_id, name_id, confidence):
        """認識バッファの更新（安定した認識のため）"""
        if face_id not in self.recognition_buffer:
            self.recognition_buffer[face_id] = []
        
        self.recognition_buffer[face_id].append((name_id, confidence))
        
        # バッファサイズを制限
        if len(self.recognition_buffer[face_id]) > self.buffer_threshold:
            self.recognition_buffer[face_id].pop(0)
        
        # 安定した認識をチェック
        if len(self.recognition_buffer[face_id]) >= self.buffer_threshold:
            # 最も多く認識された名前を取得
            names = [item[0] for item in self.recognition_buffer[face_id]]
            most_common_name = max(set(names), key=names.count)
            
            # 同じ名前が閾値以上認識された場合
            if names.count(most_common_name) >= self.buffer_threshold - 1:
                avg_confidence = sum(item[1] for item in self.recognition_buffer[face_id] 
                                   if item[0] == most_common_name) / names.count(most_common_name)
                return most_common_name, avg_confidence
        
        return None, None
    
    def draw_face_info(self, frame, x, y, w, h, name_id, confidence, is_present=False):
        """顔情報の描画"""
        name, student_id = self.parse_name_id(name_id)
        
        # 色の設定
        if name_id == "不明_000":
            color = (0, 0, 255)  # 赤: 不明
            status_text = "不明"
        elif is_present:
            color = (0, 255, 255)  # 黄: 既に出席済み
            status_text = "出席済み"
        else:
            color = (0, 255, 0)  # 緑: 認識成功
            status_text = "認識"
        
        # 顔の枠を描画
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # テキスト情報
        main_text = f"{name}({student_id})"
        conf_text = f"信頼度: {confidence:.1f}"
        
        # テキストの背景
        text_size = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x, y-60), (x + max(text_size[0], 200), y), color, -1)
        
        # メインテキスト
        cv2.putText(frame, main_text, (x+5, y-35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # 信頼度とステータス
        cv2.putText(frame, conf_text, (x+5, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # ステータス表示
        cv2.putText(frame, status_text, (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def run(self):
        """メインループの実行"""
        print("🟢 出席認識開始... 'q'キーで終了, 'r'キーで出席状況リロード, 's'キーで出席者一覧表示")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ カメラからの読み込みに失敗")
                    break
                
                # グレースケール変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 顔検出
                faces = self.detector.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
                
                # 各顔を処理
                for i, (x, y, w, h) in enumerate(faces):
                    face_roi = gray[y:y+h, x:x+w]
                    
                    # 顔認識
                    face_resized = cv2.resize(face_roi, (200, 200))
                    label, confidence = self.model.predict(face_resized)
                    
                    # ラベルから名前_IDを取得
                    name_id = self.label_map.get(label, "不明_000")
                    
                    # 信頼度チェック
                    if confidence > self.confidence_threshold:
                        name_id = "不明_000"
                    
                    # 安定した認識をチェック
                    stable_name_id, stable_confidence = self.update_recognition_buffer(
                        f"face_{i}", name_id, confidence
                    )
                    
                    # 出席記録（安定した認識のみ）
                    if stable_name_id and stable_name_id != "不明_000":
                        self.record_attendance(stable_name_id, stable_confidence)
                    
                    # 表示用の情報
                    display_name_id = name_id
                    display_confidence = confidence
                    is_present = name_id in self.today_attendance
                    
                    # 顔情報を描画
                    self.draw_face_info(frame, x, y, w, h, display_name_id, 
                                      display_confidence, is_present)
                
                # 情報表示
                info_text = f"本日の出席者: {len(self.today_attendance)}人"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # フレーム表示
                cv2.imshow("学生出席確認システム", frame)
                
                # キー処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("🔄 出席状況をリロード中...")
                    self.load_today_attendance()
                elif key == ord('s'):
                    self.show_attendance_summary()
        
        except KeyboardInterrupt:
            print("\n🛑 ユーザーによって停止されました")
        except Exception as e:
            print(f"❌ システムエラー: {e}")
        finally:
            self.cleanup()
    
    def show_attendance_summary(self):
        """出席者一覧の表示"""
        print("\n" + "="*50)
        print("📊 本日の出席者一覧")
        print("="*50)
        
        if not self.today_attendance:
            print("まだ出席者がいません")
        else:
            for i, name_id in enumerate(sorted(self.today_attendance), 1):
                name, student_id = self.parse_name_id(name_id)
                print(f"{i:2d}. {name} (学籍番号: {student_id})")
        
        print("="*50)
        print(f"合計出席者数: {len(self.today_attendance)}人\n")
    
    def cleanup(self):
        """リソースのクリーンアップ"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # 最終的な出席一覧を表示
        self.show_attendance_summary()
        print("🧹 システム終了")

def main():
    """メイン関数"""
    try:
        # 出席システムの初期化と実行
        attendance_system = StudentAttendanceSystem(
            confidence_threshold=100  # 必要に応じて調整（低いほど厳格）
        )
        attendance_system.run()
        
    except Exception as e:
        print(f"❌ システムの開始に失敗: {e}")
        print("必要なファイルを確認してください:")
        print("1. face_model.yml (訓練済みモデル)")
        print("2. labels.pkl (ラベルマップ)")
        print("3. 動作するWebカメラ")

if __name__ == "__main__":
    main()