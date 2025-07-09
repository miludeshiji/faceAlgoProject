<template>
  <el-card class="page-card" header="人脸 1:1 比对">
    <div class="content-container">
      <div class="upload-section">
        <div class="upload-pair">
          <div class="upload-item">
            <h3 class="upload-title">图像 1</h3>
            <el-upload
              class="upload-demo"
              drag
              action=""
              :auto-upload="false"
              :on-change="(file: any) => handleFileChange(file, 1)"
              :show-file-list="false"
              accept="image/*"
            >
              <el-icon class="el-icon--upload"><Picture /></el-icon>
              <div class="el-upload__text">拖拽或<em>点击上传</em></div>
            </el-upload>
            <transition name="fade">
              <el-image v-if="imageUrl1" :src="imageUrl1" class="preview-image" fit="contain" lazy />
              <div v-else class="placeholder">待上传</div>
            </transition>
          </div>

          <div class="compare-icon">
            <el-icon :size="40"><Switch /></el-icon>
          </div>

          <div class="upload-item">
            <h3 class="upload-title">图像 2</h3>
            <el-upload
              class="upload-demo"
              drag
              action=""
              :auto-upload="false"
              :on-change="(file: any) => handleFileChange(file, 2)"
              :show-file-list="false"
              accept="image/*"
            >
              <el-icon class="el-icon--upload"><Picture /></el-icon>
              <div class="el-upload__text">拖拽或<em>点击上传</em></div>
            </el-upload>
            <transition name="fade">
              <el-image v-if="imageUrl2" :src="imageUrl2" class="preview-image" fit="contain" lazy />
              <div v-else class="placeholder">待上传</div>
            </transition>
          </div>
        </div>

        <transition name="fade-slow">
          <el-button
            v-if="imageUrl1 && imageUrl2"
            type="primary"
            class="process-btn"
            :loading="loading"
            @click="compareFaces"
            size="large"
          >
            <el-icon v-if="!loading" class="el-icon--left"><DataAnalysis /></el-icon>
            <el-icon v-else class="is-loading"><Loading /></el-icon>
            开始比对
          </el-button>
        </transition>
      </div>

      <transition name="fade">
        <div class="result-section" v-if="showResult">
          <el-card class="result-card" shadow="never">
            <div slot="header" class="result-header">
              <h3>比对结果</h3>
            </div>
            <div class="result-content">
              <el-progress
                type="dashboard"
                :percentage="animatedScore"
                :color="progressColors"
                :width="200"
                :stroke-width="15"
              >
                <template #default="{ percentage }">
                  <span class="percentage-value">{{ percentage.toFixed(2) }}%</span>
                  <span class="percentage-label">相似度</span>
                </template>
              </el-progress>
              <div class="result-conclusion">
                <el-tag :type="similarityScore >= threshold ? 'success' : 'danger'" size="large" effect="dark">
                  {{ similarityScore >= threshold ? '判定为同一人' : '判定为不同人' }}
                </el-tag>
              </div>
               <div class="threshold-info">
                <span>判定阈值: {{ threshold.toFixed(2) }}</span>
                <el-slider
                  v-model="threshold"
                  :min="0"
                  :max="1"
                  :step="0.01"
                  style="width: 100%;"
                />
              </div>
            </div>
          </el-card>
        </div>
      </transition>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';
import axios from 'axios';
import { Picture, Switch, DataAnalysis, Loading } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import { gsap } from 'gsap';

const imageUrl1 = ref('');
const imageUrl2 = ref('');
const file1 = ref<File | null>(null);
const file2 = ref<File | null>(null);
const loading = ref(false);
const similarityScore = ref(0);
const animatedScore = ref(0);
const showResult = ref(false);
const threshold = ref(0.65);

const progressColors = [
  { color: '#f56c6c', percentage: 50 },
  { color: '#e6a23c', percentage: 70 },
  { color: '#5cb87a', percentage: 100 },
];

const handleFileChange = (uploadFile: { raw: File }, index: number) => {
  if (index === 1) {
    file1.value = uploadFile.raw;
    if (file1.value) imageUrl1.value = URL.createObjectURL(file1.value);
  } else {
    file2.value = uploadFile.raw;
    if (file2.value) imageUrl2.value = URL.createObjectURL(file2.value);
  }
  showResult.value = false;
};

watch(similarityScore, (newValue) => {
  gsap.to(animatedScore, { duration: 1.5, value: newValue * 100, ease: 'power3.out' });
});

const compareFaces = async () => {
  if (!file1.value || !file2.value) {
    ElMessage.warning('请上传两张人脸图像');
    return;
  }
  loading.value = true;
  showResult.value = false;

  try {
    const formData = new FormData();
    formData.append('file1', file1.value);
    formData.append('file2', file2.value);

    const response = await axios.post('/api/compare', formData);

    if (response.data.status === 'success') {
      similarityScore.value = response.data.similarity;
      showResult.value = true;
      ElMessage.success('人脸比对成功');
    } else {
      ElMessage.error(`比对失败: ${response.data.message}`);
    }
  } catch (error: any) {
    console.error('人脸比对失败:', error);
    ElMessage.error('比对时发生错误，请重试: ' + (error.response?.data?.message || error.message));
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
/* Scoped styles */
.content-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.upload-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}

.upload-pair {
  display: flex;
  justify-content: center;
  align-items: flex-start;
  gap: 20px;
  width: 100%;
}

.upload-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 16px;
  width: 45%;
}

.upload-title {
  font-weight: 500;
  margin: 0;
}

.compare-icon {
  margin-top: 80px; /* Align with uploads */
  color: var(--el-color-primary);
}

.preview-image, .placeholder {
  width: 100%;
  height: 300px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  background-color: #000;
}

.process-btn {
  margin-top: 10px;
}

.result-section {
  margin-top: 20px;
}

.result-card {
  width: 100%;
  max-width: 800px;
  margin: 0 auto;
  border: none;
  background: radial-gradient(circle, rgba(41, 55, 71, 0.5) 0%, rgba(20,20,20,0) 70%), var(--card-bg-color);
}

.result-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  padding: 20px 0;
}

.percentage-value {
  display: block;
  font-size: 40px;
  font-weight: bold;
  color: var(--text-primary-color);
}

.percentage-label {
  display: block;
  font-size: 14px;
  color: var(--text-secondary-color);
  margin-top: 5px;
}

.result-conclusion {
  margin-top: 10px;
}

.threshold-info {
  width: 60%;
  max-width: 300px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  align-items: center;
  color: var(--text-secondary-color);
  margin-top: 10px;
}

/* Transitions */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.5s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}
.fade-slow-enter-active, .fade-slow-leave-active {
  transition: opacity 0.8s ease, transform 0.8s ease;
  transform-origin: bottom;
}
.fade-slow-enter-from, .fade-slow-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

@media (max-width: 768px) {
  .upload-pair {
    flex-direction: column;
    align-items: center;
  }
  .upload-item {
    width: 90%;
  }
  .compare-icon {
    transform: rotate(90deg);
    margin-top: 0;
  }
}
</style>