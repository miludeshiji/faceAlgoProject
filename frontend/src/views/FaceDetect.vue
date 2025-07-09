<template>
  <el-card class="page-card" header="人脸检测">
    <div class="content-container">
      <!-- Upload Section -->
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
            @click="detectFaces"
            size="large"
          >
            <el-icon v-if="!loading" class="el-icon--left"><Search /></el-icon>
            <el-icon v-else class="is-loading"><Loading /></el-icon>
            检测人脸
          </el-button>
        </transition>
      </div>

      <!-- Result Section -->
      <transition name="fade">
        <div class="result-section" v-if="imageUrl">
          <!-- Image Container -->
          <div class="image-container">
            <h3>检测结果图</h3>
            <div class="image-wrapper">
              <el-image
                :src="annotatedImageUrl || imageUrl"
                class="preview-image"
                fit="contain"
              />
              <div v-if="loading" class="loading-overlay">
                <el-icon class="is-loading" :size="40"><Loading /></el-icon>
              </div>
            </div>
          </div>

          <!-- Detection Info Table (Re-added) -->
          <transition name="fade">
            <div class="detection-info" v-if="detectionResults.length > 0">
              <h3>详细信息</h3>
              <el-table :data="detectionResults" border size="small" style="width: 100%">
                <el-table-column prop="index" label="序号" width="60" align="center" />
                <el-table-column prop="score" label="置信度" width="100" align="center">
                  <template #default="scope">{{ scope.row.score.toFixed(4) }}</template>
                </el-table-column>
                <el-table-column prop="position" label="边界框位置 (x1, y1, x2, y2)" align="center">
                  <template #default="scope">
                    {{ `(${scope.row.box[0]}, ${scope.row.box[1]}), (${scope.row.box[2]}, ${scope.row.box[3]})` }}
                  </template>
                </el-table-column>
              </el-table>
            </div>
          </transition>
        </div>
      </transition>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import { UploadFilled, Search, Loading } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';

const imageUrl = ref('');
const annotatedImageUrl = ref('');
const loading = ref(false);
const file = ref<File | null>(null);
// Re-add the ref for table data
const detectionResults = ref<any[]>([]);

const handleFileChange = (uploadFile: any) => {
  file.value = uploadFile.raw;
  if (file.value) {
    imageUrl.value = URL.createObjectURL(file.value);
  }
  annotatedImageUrl.value = '';
  // Reset table data on new file upload
  detectionResults.value = [];
};

const detectFaces = async () => {
  if (!file.value) {
    ElMessage.warning('请先上传图像');
    return;
  }
  loading.value = true;
  annotatedImageUrl.value = '';
  detectionResults.value = [];

  try {
    const formData = new FormData();
    formData.append('file', file.value);
    
    // Call the backend API that returns both image and data
    const response = await axios.post('/api/detect_and_draw', formData);

    if (response.data.status === 'success') {
      const faces = response.data.faces;
      annotatedImageUrl.value = response.data.annotated_image;

      if (faces && faces.length > 0) {
        ElMessage.success(`处理成功！检测到 ${faces.length} 个人脸。`);
        // Add an index to each face object for the table
        detectionResults.value = faces.map((face: any, index: number) => ({
          ...face,
          index: index + 1
        }));
      } else {
        ElMessage.info('未检测到人脸。');
      }
    } else {
      ElMessage.error(`处理失败: ${response.data.message}`);
    }
  } catch (error: any) {
    console.error('人脸检测与绘制失败:', error);
    ElMessage.error(`处理时发生错误: ${error.response?.data?.message || '请重试'}`);
  } finally {
    loading.value = false;
  }
};
</script>

<style scoped>
.content-container { display: flex; flex-direction: column; gap: 24px; }
.upload-section { display: flex; flex-direction: column; align-items: center; gap: 20px; padding: 24px; border-radius: 8px; }
.process-btn { margin-top: 10px; }

/* Use Flexbox to arrange image and table vertically */
.result-section { 
  display: flex;
  flex-direction: column;
  gap: 24px; /* Space between image and table */
  align-items: center;
}
.image-container, .detection-info { 
  display: flex;
  flex-direction: column;
  gap: 16px;
  width: 100%;
  max-width: 900px;
}
.image-wrapper { 
  position: relative;
  width: 100%;
  line-height: 0;
}
.preview-image { 
  width: 100%;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  background-color: #000;
  min-height: 300px;
}
.loading-overlay { 
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 10;
  border-radius: 8px;
}
h3 { text-align: center; margin: 0; font-weight: 500; }

/* Transitions */
.fade-enter-active, .fade-leave-active { transition: opacity 0.5s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
.fade-slow-enter-active, .fade-slow-leave-active { transition: opacity 0.8s ease; }
.fade-slow-enter-from, .fade-slow-leave-to { opacity: 0; }
</style>
