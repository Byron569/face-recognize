import { useState } from 'react';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Layout, Menu, Typography, Grid, Button } from 'antd';
import {
  VideoCameraOutlined,
  UserOutlined,
  FileTextOutlined,
  SettingOutlined,
  DashboardOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
} from '@ant-design/icons';
import { menuConfig, themeConfig } from '../config';

const { Sider, Content, Header } = Layout;
const { Text } = Typography;

/** 菜单图标映射(按 menuConfig.icon 字符串解析)。 */
const iconMap: Record<string, React.ReactNode> = {
  video: <VideoCameraOutlined />,
  user: <UserOutlined />,
  file: <FileTextOutlined />,
  setting: <SettingOutlined />,
};

const menuItems = menuConfig.map((entry) => ({
  key: entry.key,
  icon: iconMap[entry.icon] ?? null,
  label: entry.label,
}));

export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const screens = Grid.useBreakpoint();
  const isMobile = !screens.lg; // < 992px 视为移动端
  const [collapsed, setCollapsed] = useState(false);

  const current = menuConfig.find((m) => m.key === location.pathname);

  return (
    <ConfigProvider theme={themeConfig} locale={zhCN}>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider
          theme="dark"
          width={220}
          collapsedWidth={isMobile ? 0 : 80}
          breakpoint="lg"
          collapsed={collapsed}
          onBreakpoint={(broken) => setCollapsed(broken)}
          onCollapse={(v) => setCollapsed(v)}
          zeroWidthTriggerStyle={{ top: 56 }}
          style={{
            // 移动端:侧边栏悬浮抽屉式(展开覆盖内容,不挤压);桌面:正常文档流
            position: isMobile ? 'fixed' : 'relative',
            height: '100vh',
            zIndex: 100,
            boxShadow:
              isMobile && !collapsed ? '2px 0 12px rgba(0,0,0,0.45)' : '2px 0 8px rgba(0,0,0,0.15)',
          }}
        >
          <div
            style={{
              height: 64,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 10,
              borderBottom: '1px solid rgba(255,255,255,0.1)',
            }}
          >
            <DashboardOutlined style={{ fontSize: 24, color: '#1677ff' }} />
            <Text strong style={{ color: '#fff', fontSize: 18, letterSpacing: 1 }}>
              AI Monitor
            </Text>
          </div>
          <Menu
            theme="dark"
            mode="inline"
            selectedKeys={[location.pathname]}
            items={menuItems}
            onClick={({ key }) => {
              navigate(key);
              if (isMobile) setCollapsed(true); // 手机上选完自动收起
            }}
            style={{ marginTop: 8 }}
          />
        </Sider>
        <Layout>
          <Header
            style={{
              background: '#fff',
              padding: '0 16px',
              height: 48,
              lineHeight: '48px',
              borderBottom: '1px solid #f0f0f0',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <Button
              type="text"
              size="small"
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed(!collapsed)}
              style={{ fontSize: 16 }}
            />
            <Text strong style={{ fontSize: 16, color: '#1a1a1a' }}>
              {current ? current.label : ''}
            </Text>
          </Header>
          <Content
            style={{
              padding: isMobile ? 8 : 24,
              background: '#f0f2f5',
              minHeight: 'calc(100vh - 48px)',
            }}
          >
            <Outlet />
          </Content>
        </Layout>
      </Layout>
    </ConfigProvider>
  );
}
