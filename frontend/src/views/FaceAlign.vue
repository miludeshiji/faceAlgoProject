<template>
  <el-card class="page-card" header="人脸对齐">
    <div class="content-container">
      <div class="upload-section">
        <el-upload
          class="upload-demo"
          drag
          action=""
          :auto-upload="false"
          :on-change="handleFileChange"
          :show-file-list="false"
          accept="image/*"
        >
          <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
          <div class="el-upload__text">拖放图片到此处，或<em>点击上传</em></div>
        </el-upload>
        <transition name="fade-slow">
          <el-button
            v-if="imageUrl"
            type="primary"
            class="process-btn"
            :loading="loading"
            @click="processImage"
            size="large"
          >
            <el-icon v-if="!loading" class="el-icon--left"><MagicStick /></el-icon>
            <el-icon v-else class="is-loading"><Loading /></el-icon>
            处理图像
          </el-button>
        </transition>
      </div>

      <transition name="fade">
        <el-row :gutter="20" class="result-section" v-if="imageUrl">
          <el-col :md="12" :xs="24" class="image-col">
            <div class="image-container">
              <h3>原始图像</h3>
              <el-image
                v-if="imageUrl"
                :src="imageUrl"
                class="preview-image"
                fit="contain"
                lazy
              />
            </div>
          </el-col>

          <el-col :md="12" :xs="24" class="image-col">
            <div class="image-container">
              <h3>对齐结果</h3>
              <div class="placeholder-wrapper">
                <div v-if="loading && alignedImages.length === 0" class="placeholder">
                  <el-icon class="is-loading" :size="30"><Loading /></el-icon>
                  <p>正在处理中...</p>
                </div>
                <div v-else-if="!loading && alignedImages.length === 0 && file" class="placeholder">
                  处理后在此显示结果
                </div>
                 <div v-else-if="!file" class="placeholder">
                  请先上传图像
                </div>
                <transition-group tag="div" name="list" class="aligned-results-grid">
                  <el-image
                    v-for="(img, index) in alignedImages"
                    :key="img + index"
                    :src="img"
                    class="aligned-image"
                    fit="cover"
                    lazy
                  />
                </transition-group>
              </div>
            </div>
          </el-col>
        </el-row>
      </transition>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import { UploadFilled, MagicStick, Loading } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';
import type { UploadFile } from 'element-plus';

const imageUrl = ref('');
const alignedImages = ref<string[]>([]);
const loading = ref(false);
const file = ref<File | null>(null);

const handleFileChange = (uploadFile: UploadFile) => {
  if (uploadFile.raw) {
    file.value = uploadFile.raw;
    imageUrl.value = URL.createObjectURL(file.value);
    alignedImages.value = [];
  }
};

const processImage = async () => {
  if (!file.value) {
    ElMessage.warning('请先上传图像');
    return;
  }
  loading.value = true;
  alignedImages.value = [];

  try {
    const detectForm = new FormData();
    detectForm.append('file', file.value);
    const detectResponse = await axios.post('/api/detect', detectForm);

    const faces = detectResponse.data.faces;
    if (detectResponse.data.status !== 'success' || !faces || faces.length === 0) {
      ElMessage.error('未检测到人脸，请上传包含清晰人脸的图像');
      loading.value = false;
      return;
    }

    let successCount = 0;
    // To show images one by one, we don't use Promise.all to wait for all
    for (const face of faces) {
        try {
            const alignForm = new FormData();
            alignForm.append('file', file.value as Blob);
            alignForm.append('landmarks', JSON.stringify(face.landmarks));
            const alignResponse = await axios.post('/api/align', alignForm);

            if (alignResponse.data.status === 'success') {
                successCount++;
                alignedImages.value.push(alignResponse.data.aligned_image);
            } else {
                console.error(`一个多余人脸对齐失败: ${alignResponse.data.message}`);
            }
        } catch (err) {
            console.error('单个对齐请求失败:', err);
        }
    }

    if (successCount > 0) {
      ElMessage.success(`处理完成！成功对齐 ${successCount} / ${faces.length} 个人脸。`);
    } else {
      ElMessage.error('所有的人脸都对齐失败，请检查后端服务日志。');
    }

  } catch (error: any) {
    console.error('处理图像失败:', error.response?.data || error.message || error);
    ElMessage.error(`处理图像时发生错误: ${error.response?.data?.message || error.message || '未知错误'}`);
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
/* Main Content */
.content-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

/* Upload Section */
.upload-section {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
  padding: 24px;
  border-radius: 8px;
}

.process-btn {
  margin-top: 10px;
}

/* Results */
.result-section {
  width: 100%;
}

.image-col {
  margin-bottom: 20px;
}

.image-container {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.image-container h3 {
  text-align: center;
  margin-bottom: 16px;
  font-weight: 500;
}

.preview-image {
  width: 100%;
  min-height: 400px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  background-color: #000;
}

.placeholder-wrapper {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  min-height: 400px;
  background-color: #000;
  padding: 10px;
}

.placeholder {
  flex-grow: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-secondary-color);
  font-size: 16px;
  border-radius: 8px;
  border: none;
}

.aligned-results-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-start;
  align-content: flex-start;
  overflow-y: auto;
  flex-grow: 1;
}

.aligned-image {
  width: 112px;
  height: 112px;
  border-radius: 6px;
  border: 1px solid var(--border-color);
  background-color: var(--card-bg-color);
  transition: transform 0.3s ease;
}
.aligned-image:hover {
  transform: scale(1.1);
  z-index: 2;
  border-color: var(--el-color-primary);
}


/* Transitions */
.fade-enter-active, .fade-leave-active {
  transition: opacity 0.5s ease;
}
.fade-enter-from, .fade-leave-to {
  opacity: 0;
}

.fade-slow-enter-active, .fade-slow-leave-active {
  transition: opacity 0.8s ease;
}
.fade-slow-enter-from, .fade-slow-leave-to {
  opacity: 0;
}

.list-enter-active,
.list-leave-active {
  transition: all 0.5s ease;
}
.list-enter-from,
.list-leave-to {
  opacity: 0;
  transform: scale(0.3);
}
</style>