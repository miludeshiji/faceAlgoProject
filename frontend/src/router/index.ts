import { createRouter, createWebHistory } from 'vue-router'
import { DefineComponent } from 'vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      redirect: '/align'
    },
    {
      path: '/align',
      name: 'faceAlign',
      component: () => import('../views/FaceAlign.vue') as Promise<{ default: DefineComponent }>,
      meta: { title: '人脸对齐' }
    },
    {
      path: '/detect',
      name: 'faceDetect',
      component: () => import('../views/FaceDetect.vue') as Promise<{ default: DefineComponent }>,
      meta: { title: '人脸检测' }
    },
    {
      path: '/register',
      name: 'faceRegister',
      component: () => import('../views/FaceRegister.vue') as Promise<{ default: DefineComponent }>,
      meta: { title: '人脸注册' }
    },
    {
      path: '/compare',
      name: 'faceCompare',
      component: () => import('../views/FaceCompare.vue') as Promise<{ default: DefineComponent }>,
      meta: { title: '人脸比对' }
    },
    {
      path: '/search',
      name: 'faceSearch',
      component: () => import('../views/FaceSearch.vue') as Promise<{ default: DefineComponent }>,
      meta: { title: '人脸搜索' }
    },
    {
      path: '/monitoring',
      name: 'Monitoring',
      component: () => import('../views/Monitoring.vue') as Promise<{ default: DefineComponent }>,
      meta: { title: '监控' }
    },
    {
      path: '/person-management',
      name: 'PersonManagement',
      component: () => import('../views/PersonManagement.vue') as Promise<{ default: DefineComponent }>,
      meta: { title: '人员管理' }
    }
  ]
})

export default router