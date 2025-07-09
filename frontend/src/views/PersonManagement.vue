<template>
  <el-card class="page-card" header="人员库管理">
    <div class="toolbar">
      <el-button @click="getPersons" :loading="loading" type="primary">
        <el-icon class="el-icon--left"><Refresh /></el-icon>
        刷新列表
      </el-button>
      <el-alert
        title="在此页面，您可以查看和删除已在系统中注册的所有人员信息。"
        type="info"
        show-icon
        :closable="false"
        style="width: 100%;"
      />
    </div>
    
    <el-table :data="personList" v-loading="loading" style="width: 100%" border>
      <el-table-column type="index" label="序号" width="80" align="center" />
      <el-table-column prop="id" label="人员ID" />
      <el-table-column label="操作" width="120" align="center">
        <template #default="scope">
          <el-popconfirm
            title="您确定要从数据库中删除此人吗？此操作不可撤销。"
            confirm-button-text="确定"
            cancel-button-text="取消"
            @confirm="handleDelete(scope.row.id)"
          >
            <template #reference>
              <el-button type="danger" size="small">
                <el-icon class="el-icon--left"><Delete /></el-icon>
                删除
              </el-button>
            </template>
          </el-popconfirm>
        </template>
      </el-table-column>
       <template #empty>
        <el-empty description="人员库为空，请先在'人脸注册'页面添加人员" />
      </template>
    </el-table>
  </el-card>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { ElMessage } from 'element-plus';
import { Delete, Refresh } from '@element-plus/icons-vue';

interface Person {
  id: string;
}

const loading = ref(false);
const personList = ref<Person[]>([]);

// 获取所有人员列表
const getPersons = async () => {
  loading.value = true;
  try {
    const response = await axios.get('/api/persons');
    if (response.data.status === 'success') {
      // 将返回的ID字符串数组转换为表格需要的对象数组
      personList.value = response.data.persons.map((id: string) => ({ id }));
    } else {
      ElMessage.error(`获取列表失败: ${response.data.message}`);
    }
  } catch (error: any) {
    ElMessage.error(`API请求错误: ${error.response?.data?.message || error.message}`);
  } finally {
    loading.value = false;
  }
};

// 处理删除操作
const handleDelete = async (personId: string) => {
  try {
    const response = await axios.delete(`/api/persons/${personId}`);
    if (response.data.status === 'success') {
      ElMessage.success(response.data.message);
      // 删除成功后，重新获取列表以刷新UI
      await getPersons();
    } else {
      ElMessage.error(`删除失败: ${response.data.message}`);
    }
  } catch (error: any) {
    ElMessage.error(`API请求错误: ${error.response?.data?.message || error.message}`);
  }
};

// 组件挂载时自动加载数据
onMounted(() => {
  getPersons();
});
</script>

<style scoped>
.toolbar {
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
  align-items: center;
}
</style>