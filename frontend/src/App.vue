<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="logo-container">
        <el-icon :size="24" style="margin-right: 10px;"><Cpu /></el-icon>
        <h1 class="header-title">AI 人脸算法平台</h1>
      </div>
    </el-header>
    <el-container>
      <el-aside width="200px" class="app-sidebar">
        <el-menu
          :default-active="$route.path"
          class="el-menu-vertical-demo"
          background-color="transparent"
          text-color="rgba(255, 255, 255, 0.7)"
          active-text-color="#409EFF"
          router
        >
          <el-menu-item index="/align">
            <el-icon><MagicStick /></el-icon>
            <span>人脸对齐</span>
          </el-menu-item>
          <el-menu-item index="/detect">
            <el-icon><View /></el-icon>
            <span>人脸检测</span>
          </el-menu-item>
          <el-menu-item index="/register">
            <el-icon><User /></el-icon>
            <span>人脸注册</span>
          </el-menu-item>
          <el-menu-item index="/compare">
            <el-icon><Files /></el-icon>
            <span>人脸比对</span>
          </el-menu-item>
          <el-menu-item index="/search">
            <el-icon><Search /></el-icon>
            <span>人脸搜索</span>
          </el-menu-item>
          <el-menu-item index="/monitoring">
            <el-icon><VideoCamera /></el-icon>
            <span>安防监控</span>
          </el-menu-item>
          <el-menu-item index="/person-management">
            <el-icon><Tickets /></el-icon>
            <span>人员库管理</span>
          </el-menu-item>
        </el-menu>
      </el-aside>
      <el-main class="app-main">
        <router-view v-slot="{ Component }">
          <transition name="fade" mode="out-in">
            <component :is="Component" />
          </transition>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
// Import more distinct and appropriate icons
import {
  MagicStick,
  View,
  User,
  Files,
  Search,
  Cpu,
  VideoCamera,
  Tickets
} from '@element-plus/icons-vue'
</script>

<style>
/* Global Styles for Dark Theme */
:root {
  --app-bg-color: #141414;
  --sidebar-bg-color: #1d1d1d;
  --header-bg-color: #1d1d1d;
  --card-bg-color: #262626;
  --card-header-color: #1d1d1d;
  --text-primary-color: #ffffff;
  --text-secondary-color: #a8a8a8;
  --border-color: #363636;
  --el-color-primary: #409EFF;
}

body {
  background-color: var(--app-bg-color);
  color: var(--text-primary-color);
  margin: 0;
  font-family: "Helvetica Neue", Helvetica, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "微软雅黑", Arial, sans-serif;
}

/* Overriding Element Plus variables for dark mode */
.el-card {
  background-color: var(--card-bg-color) !important;
  border: 1px solid var(--border-color) !important;
}

.el-card__header {
  background-color: var(--card-header-color) !important;
  color: var(--text-primary-color) !important;
  border-bottom: 1px solid var(--border-color) !important;
  font-weight: bold;
}

.el-upload-dragger {
  background-color: rgba(255, 255, 255, 0.05) !important;
  border: 2px dashed var(--border-color) !important;
  transition: border-color 0.3s, background-color 0.3s;
}

.el-upload-dragger:hover {
    border-color: var(--el-color-primary) !important;
    background-color: rgba(64, 158, 255, 0.1) !important;
}

.el-upload__text {
  color: var(--text-secondary-color) !important;
}

.el-upload__text em {
  color: var(--el-color-primary) !important;
  font-style: normal;
}

.el-table {
  background-color: var(--card-bg-color) !important;
}

.el-table th, .el-table tr, .el-table td {
  background-color: transparent !important;
  color: var(--text-secondary-color) !important;
  border-bottom: 1px solid var(--border-color) !important;
}

.el-table__header-wrapper th {
   color: var(--text-primary-color) !important;
}

.el-form-item__label {
  color: var(--text-secondary-color) !important;
}

.el-input__wrapper {
  background-color: #303030 !important;
  box-shadow: none !important;
  border: 1px solid var(--border-color) !important;
}

.el-input__inner {
  color: var(--text-primary-color) !important;
}

.placeholder, .info-placeholder {
  border: 2px dashed var(--border-color) !important;
  color: var(--text-secondary-color) !important;
  background-color: rgba(0,0,0,0.2);
  border-radius: 8px;
}

.el-skeleton {
  --el-skeleton-color: #3a3a3a !important;
  --el-skeleton-to-color: #4a4a4a !important;
}

h3 {
  color: var(--text-primary-color);
}
</style>

<style scoped>
.app-container {
  height: 100vh;
  background-color: var(--app-bg-color);
}

.app-header {
  background-color: var(--header-bg-color);
  color: white;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid var(--border-color);
  position: sticky;
  top: 0;
  z-index: 100;
}

.logo-container {
  display: flex;
  align-items: center;
}

.header-title {
  margin: 0;
  font-size: 20px;
  font-weight: 500;
}

.app-sidebar {
  background-color: var(--sidebar-bg-color);
  height: calc(100vh - 61px);
  position: sticky;
  top: 61px;
  border-right: 1px solid var(--border-color);
}

.el-menu-item {
  transition: background-color 0.3s;
}

.el-menu-item:hover {
  background-color: rgba(255, 255, 255, 0.05) !important;
}

.el-menu-vertical-demo {
    border-right: none;
}

.app-main {
  padding: 20px;
  overflow-y: auto;
  background-color: var(--app-bg-color);
}

/* Router view transition */
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>