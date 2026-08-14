import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { Layout, Menu, Typography } from 'antd';
import {
  VideoCameraOutlined,
  UserOutlined,
  FileTextOutlined,
  SettingOutlined,
  DashboardOutlined,
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

  const current = menuConfig.find((m) => m.key === location.pathname);

  return (
    <ConfigProvider theme={themeConfig} locale={zhCN}>
      <Layout style={{ minHeight: '100vh' }}>
        <Sider
          theme="dark"
          width={220}
          style={{
            boxShadow: '2px 0 8px rgba(0,0,0,0.15)',
            zIndex: 10,
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
            onClick={({ key }) => navigate(key)}
            style={{ marginTop: 8 }}
          />
        </Sider>
        <Layout>
          <Header
            style={{
              background: '#fff',
              padding: '0 24px',
              height: 48,
              lineHeight: '48px',
              borderBottom: '1px solid #f0f0f0',
              display: 'flex',
              alignItems: 'center',
            }}
          >
            <Text strong style={{ fontSize: 16, color: '#1a1a1a' }}>
              {current ? current.label : ''}
            </Text>
          </Header>
          <Content
            style={{
              padding: 24,
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
