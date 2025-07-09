<template>
  <el-card class="page-card" header="人脸 1:N 搜索">
    <div class="content-container">
      <!-- Search Section -->
      <div class="search-section">
        <div class="upload-area">
          <h3>待搜索图像</h3>
          <el-upload
            class="upload-demo" drag action="" :auto-upload="false"
            :on-change="handleFileChange" :show-file-list="false" accept="image/*"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">拖放图片到此处，或<em>点击上传</em></div>
          </el-upload>
        </div>

        <div class="preview-area">
          <h3>图像预览</h3>
          <transition name="fade" mode="out-in">
            <el-image v-if="imageUrl" :src="imageUrl" class="preview-image" fit="contain" lazy />
            <div v-else class="placeholder">上传后在此预览</div>
          </transition>
        </div>

        <div class="controls-area">
          <h3>搜索参数</h3>
          <el-form-item label="匹配阈值">
            <el-slider v-model="threshold" :min="0" :max="1" :step="0.01" />
            <span class="threshold-value">{{ threshold.toFixed(2) }}</span>
          </el-form-item>
          <el-button
            type="primary" class="search-btn" size="large"
            :loading="loading" @click="searchFace" :disabled="!imageUrl"
          >
            <el-icon v-if="!loading" class="el-icon--left"><ZoomIn /></el-icon>
            <el-icon v-else class="is-loading"><Loading /></el-icon>
            在库中搜索
          </el-button>
        </div>
      </div>

      <!-- Result Section - Modified for multiple results -->
      <transition name="fade-slow">
        <div class="result-section" v-if="showResult">
          <el-card class="result-card" shadow="never">
            <template #header>
              <div class="result-header">
                <h3>搜索到 {{ searchResults.length }} 个结果</h3>
              </div>
            </template>
            <div class="result-content-grid">
              <!-- Loop through all results -->
              <div v-for="(result, index) in searchResults" :key="index" class="result-item-card">
                <!-- Matched face -->
                <div v-if="result.id !== 'unknown'" class="match-result">
                  <el-avatar :size="80" class="result-avatar">{{ result.id }}</el-avatar>
                  <div class="result-info">
                    <p class="result-id">{{ result.id }}</p>
                    <p class="result-score">相似度: <strong>{{ result.score.toFixed(4) }}</strong></p>
                  </div>
                  <el-tag type="success" effect="dark" class="result-tag">匹配成功</el-tag>
                </div>
                <!-- Unmatched face -->
                <div v-else class="no-match-result">
                    <el-icon :size="60" color="#909399"><UserFilled /></el-icon>
                    <p class="result-id">未知</p>
                    <p class="result-score">最高相似度: {{ result.score.toFixed(4) }}</p>
                    <el-tag type="info" effect="dark" class="result-tag">未匹配</el-tag>
                </div>
              </div>
               <!-- Placeholder for no results -->
              <div v-if="searchResults.length === 0" class="no-results-placeholder">
                  <el-icon :size="80" color="#909399"><CircleCloseFilled /></el-icon>
                  <p>未在图像中检测到人脸，或库中无任何匹配项。</p>
              </div>
            </div>
          </el-card>
        </div>
      </transition>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import { UploadFilled, ZoomIn, Loading, UserFilled, CircleCloseFilled } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';

const imageUrl = ref('');
const file = ref<File | null>(null);
const loading = ref(false);
const threshold = ref(0.65);
const showResult = ref(false);
// Change state to an array to hold multiple results
const searchResults = ref<any[]>([]);

const handleFileChange = (uploadFile: any) => {
  file.value = uploadFile.raw;
  if (file.value) imageUrl.value = URL.createObjectURL(file.value);
  showResult.value = false;
  searchResults.value = []; // Clear previous results
};

const searchFace = async () => {
  if (!file.value) return ElMessage.warning('请先上传图像');
  loading.value = true;
  try {
    const formData = new FormData();
    formData.append('file', file.value);
    formData.append('threshold', threshold.value.toString());
    
    // Call the modified backend API
    const response = await axios.post('/api/search', formData);

    if (response.data.status === 'success') {
      // Assign the array of results
      searchResults.value = response.data.results;
      showResult.value = true;
      const matchCount = searchResults.value.filter(r => r.id !== 'unknown').length;
      ElMessage.success(`搜索完成！共处理了 ${searchResults.value.length} 张人脸，其中 ${matchCount} 张匹配成功。`);
    } else {
      ElMessage.error(`搜索失败: ${response.data.message}`);
    }
  } catch (error: any) {
    ElMessage.error(`搜索时发生错误: ${error.response?.data?.message || '请重试'}`);
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
h3 { text-align: center; margin-bottom: 16px; font-weight: 500; }
.content-container { display: flex; flex-direction: column; gap: 24px; }
.search-section { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 24px; align-items: flex-start; }
.upload-area, .preview-area, .controls-area { display: flex; flex-direction: column; gap: 16px; }
.preview-image, .placeholder { width: 100%; height: 350px; border-radius: 8px; border: 1px solid var(--border-color); background-color: #000; }
.el-form-item { width: 100%; display: flex; flex-direction: column; align-items: flex-start; gap: 10px; }
.el-form-item .el-slider{ width: 100%;}
.threshold-value { color: var(--text-secondary-color); min-width: 40px; }
.search-btn { width: 100%; }

.result-section { margin-top: 10px; }
.result-card { width: 100%; border: none; background-color: var(--card-bg-color); }
.result-header { text-align: center; }

/* New grid layout for results */
.result-content-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  justify-content: center;
  padding: 20px;
}

.result-item-card {
  background-color: var(--app-bg-color);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 20px;
  width: 200px;
  display: flex;
  flex-direction: column;
  align-items: center;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.result-item-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
}

.match-result, .no-match-result { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 10px; }
.result-avatar { font-size: 20px; background-color: var(--el-color-primary); }
.result-id { font-size: 20px; font-weight: bold; margin: 0; }
.result-score { font-size: 14px; color: var(--text-secondary-color); margin: 0; }
.result-tag { margin-top: 10px; }

.no-results-placeholder {
    width: 100%;
    text-align: center;
    color: var(--text-secondary-color);
    padding: 40px;
}

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.5s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
.fade-slow-enter-active, .fade-slow-leave-active { transition: opacity 0.8s ease, transform 0.5s ease; }
.fade-slow-enter-from, .fade-slow-leave-to { opacity: 0; transform: translateY(30px); }

@media (max-width: 992px) {
  .search-section { grid-template-columns: 1fr; }
  .preview-area { order: -1; }
}
</style>