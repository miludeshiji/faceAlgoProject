<template>
  <el-card class="page-card" header="人脸注册">
    <div class="content-container">
      <div class="register-form">
        <div class="form-step">
          <div class="step-header">
            <div class="step-number">1</div>
            <h3>输入身份ID</h3>
          </div>
          <el-input v-model="faceId" placeholder="例如：zhangsan_001" size="large"/>
        </div>

        <div class="form-step">
          <div class="step-header">
            <div class="step-number">2</div>
            <h3>上传人脸图像</h3>
          </div>
          <el-upload
            class="upload-demo"
            drag
            action=""
            :auto-upload="false"
            :on-change="handleFileChange"
            :show-file-list="false"
            accept="image/*"
          >
            <transition name="fade" mode="out-in">
              <div v-if="imageUrl" class="upload-preview">
                <el-image :src="imageUrl" class="preview-image" fit="cover"/>
                <div class="preview-overlay">点击或拖拽以更换</div>
              </div>
              <div v-else class="upload-placeholder">
                <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                <div class="el-upload__text">拖放图片到此处，或<em>点击上传</em></div>
              </div>
            </transition>
          </el-upload>
        </div>

        <div class="form-actions">
          <el-button
            type="primary"
            size="large"
            :loading="loading"
            @click="registerFace"
            :disabled="!faceId || !imageUrl"
          >
            <el-icon v-if="!loading" class="el-icon--left"><UserFilled /></el-icon>
            <el-icon v-else class="is-loading"><Loading /></el-icon>
            确认注册
          </el-button>
          <el-button @click="resetForm" size="large">重置信息</el-button>
        </div>
      </div>
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
import { UploadFilled, UserFilled, Loading } from '@element-plus/icons-vue';
import { ElMessage } from 'element-plus';

const faceId = ref('');
const imageUrl = ref('');
const loading = ref(false);
const file = ref<File | null>(null);

const handleFileChange = (uploadFile: any) => {
  file.value = uploadFile.raw;
  if (file.value) imageUrl.value = URL.createObjectURL(file.value);
};

const registerFace = async () => {
  if (!faceId.value || !file.value) {
    return ElMessage.warning('请输入人脸ID并上传图像');
  }

  loading.value = true;
  try {
    const formData = new FormData();
    formData.append('file', file.value);
    formData.append('id', faceId.value);
    const response = await axios.post('/api/register', formData);

    if (response.data.status === 'success') {
      ElMessage.success(response.data.message);
      resetForm();
    } else {
      ElMessage.error(`注册失败: ${response.data.message}`);
    }
  } catch (error: any) {
    ElMessage.error(`注册时发生错误: ${error.response?.data?.message || '请重试'}`);
  } finally {
    loading.value = false;
  }
};

const resetForm = () => {
  faceId.value = '';
  imageUrl.value = '';
  file.value = null;
};
</script>

<style scoped>
.content-container { display: flex; justify-content: center; padding: 20px; }
.register-form { width: 100%; max-width: 500px; display: flex; flex-direction: column; gap: 30px; }
.form-step { display: flex; flex-direction: column; gap: 15px; }
.step-header { display: flex; align-items: center; gap: 12px; }
.step-number { background-color: var(--el-color-primary); color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; }
h3 { margin: 0; font-weight: 500; }
.upload-placeholder { padding: 40px; }
.upload-preview { position: relative; width: 100%; height: 250px; }
.preview-image { width: 100%; height: 100%; border-radius: 6px; }
.preview-overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); color: white; display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity 0.3s; border-radius: 6px; cursor: pointer; }
.upload-preview:hover .preview-overlay { opacity: 1; }
.form-actions { display: flex; justify-content: center; gap: 10px; margin-top: 10px; }
.fade-enter-active, .fade-leave-active { transition: opacity 0.4s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>