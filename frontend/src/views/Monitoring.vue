<template>
  <el-card class="page-card" header="智能安防监控仪表盘 (实时视频模式)">
    <div class="control-panel">
      <el-button 
        type="primary" 
        @click="startCamera" 
        :disabled="isCameraOn"
      >
        <el-icon class="el-icon--left"><VideoCamera /></el-icon>
        开启摄像头
      </el-button>
      <el-button 
        type="danger" 
        @click="stopCamera" 
        :disabled="!isCameraOn"
      >
        <el-icon class="el-icon--left"><SwitchButton /></el-icon>
        关闭摄像头
      </el-button>
      <el-button 
        @click="toggleAnalysis" 
        :disabled="!isCameraOn"
        :type="isAnalyzing ? 'warning' : 'success'"
      >
        <el-icon v-if="isAnalyzing" class="el-icon--left is-loading"><Loading /></el-icon>
        <el-icon v-else class="el-icon--left"><CaretRight /></el-icon>
        {{ isAnalyzing ? '暂停分析' : '开始分析' }}
      </el-button>
    </div>

    <div class="monitoring-container">
      <div class="left-panel">
        <el-card shadow="never">
          <template #header><span>实时画面 / 分析结果</span></template>
          <div class="video-container">
            <video ref="videoRef" class="live-video" autoplay playsinline muted></video>

            <el-image 
              v-if="annotatedImageUrl" 
              :src="annotatedImageUrl" 
              fit="contain" 
              class="annotated-image"
            />
            
            <div v-if="loading" class="loading-overlay">
              <el-icon class="is-loading" :size="40"><Loading /></el-icon>
              <span>正在分析...</span>
            </div>

            <div v-if="cameraError || !isCameraOn" class="placeholder-info">
              <el-icon :size="50"><VideoPause /></el-icon>
              <span>{{ cameraError || '摄像头已关闭' }}</span>
            </div>
          </div>
        </el-card>
      </div>

      <div class="right-panel">
        <el-card shadow="never" class="event-log-card">
          <template #header>
             <div class="card-header">
              <span>事件日志</span>
               <el-button type="danger" size="small" @click="clearLog" plain>清空日志</el-button>
            </div>
          </template>
          <el-table :data="eventLog" height="480" style="width: 100%" empty-text="暂无事件记录">
            <el-table-column prop="time" label="时间" width="100" align="center"/>
            <el-table-column prop="personId" label="识别ID" align="center">
                 <template #default="scope">
                    <el-tag :type="scope.row.status === '允许进入' ? 'success' : 'danger'">
                      {{ scope.row.personId }}
                    </el-tag>
                </template>
            </el-table-column>
            <el-table-column prop="status" label="状态" width="120" align="center"/>
            <el-table-column prop="score" label="置信度" width="100" align="center"/>
          </el-table>
        </el-card>
      </div>
    </div>
     <canvas ref="canvasRef" style="display: none;"></canvas>
  </el-card>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue';
import axios from 'axios';
import { ElMessage, ElNotification } from 'element-plus';
import { VideoCamera, SwitchButton, Loading, CaretRight, VideoPause } from '@element-plus/icons-vue';

// --- State Management ---
const loading = ref(false);
const annotatedImageUrl = ref('');
const eventLog = ref<any[]>([]);

// Video and analysis state
const videoRef = ref<HTMLVideoElement | null>(null);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const isCameraOn = ref(false);
const isAnalyzing = ref(false);
const cameraError = ref('');
let stream: MediaStream | null = null;
let analysisTimer: number | null = null;


// --- Camera & Analysis Control ---
const startCamera = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraError.value = "错误：您的浏览器不支持摄像头访问功能。";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    if (videoRef.value) {
      videoRef.value.srcObject = stream;
      isCameraOn.value = true;
      cameraError.value = '';
      startAnalysis(); // 摄像头开启后自动开始分析
    }
  } catch (err: any) {
    console.error("摄像头启动失败:", err);
    if (err.name === 'NotAllowedError') {
      cameraError.value = '您已拒绝摄像头访问权限，请在浏览器设置中重新授权。';
    } else {
      cameraError.value = `无法访问摄像头: ${err.message}`;
    }
  }
};

const stopCamera = () => {
  stopAnalysis();
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null;
  }
  isCameraOn.value = false;
  stream = null;
  annotatedImageUrl.value = ''; // 关闭摄像头时清空上一张分析图
};

const startAnalysis = () => {
  if (isAnalyzing.value || !isCameraOn.value) return;
  isAnalyzing.value = true;
  // 每 2.5 秒分析一次
  analysisTimer = window.setInterval(captureAndAnalyze, 2500);
};

const stopAnalysis = () => {
  if (analysisTimer) {
    clearInterval(analysisTimer);
    analysisTimer = null;
  }
  isAnalyzing.value = false;
};

const toggleAnalysis = () => {
  if (isAnalyzing.value) {
    stopAnalysis();
  } else {
    startAnalysis();
  }
};

// --- Core Logic ---
const captureAndAnalyze = async () => {
  if (loading.value || !videoRef.value || !canvasRef.value) return;
  
  loading.value = true;

  // 将当前视频帧绘制到隐藏的canvas上
  const video = videoRef.value;
  const canvas = canvasRef.value;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext('2d');
  context?.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
  
  // 从canvas获取图片Blob数据
  canvas.toBlob(async (blob) => {
    if (!blob) {
      loading.value = false;
      return;
    }
    
    const formData = new FormData();
    formData.append('file', blob, 'capture.jpg');
    formData.append('threshold', '0.65');

    try {
      const response = await axios.post('/api/monitor_recognize', formData);
      if (response.data.status === 'success') {
        annotatedImageUrl.value = response.data.annotated_image;
        const events = response.data.events;
        
        if (events.length > 0) {
          const timestamp = new Date().toLocaleTimeString();
          let hasUnknownVisitor = false;
          
          events.forEach((event: any) => {
            if (event.status !== '允许进入') hasUnknownVisitor = true;
            eventLog.value.unshift({ time: timestamp, ...event });
          });

          if (hasUnknownVisitor) {
            new Audio('/alert.mp3').play().catch(e => console.error("音效播放失败:", e));
            ElNotification({
              title: '安全警报',
              message: '检测到未授权的陌生访客！',
              type: 'warning',
              duration: 5000, // 警报持续5秒
            });
          }
        }
      } else {
        ElMessage.error(`分析失败: ${response.data.message}`);
      }
    } catch (error: any) {
       console.error(`API请求错误: ${error.response?.data?.message || error.message}`);
    } finally {
       loading.value = false;
    }
  }, 'image/jpeg');
};

const clearLog = () => {
  eventLog.value = [];
  ElMessage.success('事件日志已清空。');
};

// --- Lifecycle Hook ---
// 组件卸载时，确保关闭摄像头，防止资源占用
onUnmounted(() => {
  if (isCameraOn.value) {
    stopCamera();
  }
});
</script>

<style scoped>
.control-panel {
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
}
.monitoring-container {
  display: flex;
  gap: 20px;
}
.left-panel {
  flex: 3;
}
.right-panel {
  flex: 2;
}
.video-container {
  position: relative;
  width: 100%;
  min-height: 480px;
  background-color: #000;
  border-radius: 4px;
  overflow: hidden;
}
.live-video, .annotated-image {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
}
.annotated-image {
  z-index: 2;
}
.loading-overlay, .placeholder-info {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #fff;
  z-index: 3;
}
.loading-overlay {
  background-color: rgba(0, 0, 0, 0.7);
}
.placeholder-info {
   background-color: rgba(0, 0, 0, 0.9);
   gap: 15px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.event-log-card {
  height: 540px;
}
</style>