class Result:
    """Result Monad สำหรับห่อหุ้มสถานะ สำเร็จ (Success) หรือ ล้มเหลว (Failure)"""
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error
        self.is_success = error is None

    @staticmethod
    def success(value):
        return Result(value=value)

    @staticmethod
    def failure(error):
        return Result(error=error)

    def bind(self, func):
        """ฟังก์ชันสำหรับส่งค่าต่อยอดไปยังสเต็ปถัดไป (ถ้าสถานะยังปกติอยู่)"""
        if not self.is_success:
            return self  # ถ้าพังแล้ว ให้ส่ง Error วิ่งผ่านยาวๆ ไปเลยไม่ต้องทำต่อ
        try:
            return func(self.value)
        except Exception as e:
            return Result.failure(str(e))