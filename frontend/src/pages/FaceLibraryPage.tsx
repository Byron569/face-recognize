import { useState } from 'react';
import { Card, Row, Col, Button, Input, Modal, message, Avatar, Tag, Space, Upload, Result, Descriptions, Grid } from 'antd';
import { PlusOutlined, DeleteOutlined, SearchOutlined, UserOutlined, EditOutlined, UploadOutlined, ScanOutlined, VideoCameraOutlined } from '@ant-design/icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchFaces, deleteFace, batchImportFaces, searchFace, Identity } from '../api/faces';
import FaceRegisterModal from '../components/FaceRegisterModal';
import CameraRegisterModal from '../components/CameraRegisterModal';
import FaceEditModal from '../components/FaceEditModal';

const colors = ['#1677ff', '#52c41a', '#fa8c16', '#722ed1', '#eb2f96', '#13c2c2'];

export default function FaceLibraryPage() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [registerOpen, setRegisterOpen] = useState(false);
  const [cameraRegisterOpen, setCameraRegisterOpen] = useState(false);
  const [editFace, setEditFace] = useState<Identity | null>(null);
  const [batchOpen, setBatchOpen] = useState(false);
  const [batchName, setBatchName] = useState('');
  const [batchFiles, setBatchFiles] = useState<any[]>([]);
  const [batchResult, setBatchResult] = useState<any[] | null>(null);
  const [searchOpen, setSearchOpen] = useState(false);
  const [searchFile, setSearchFile] = useState<any>(null);
  const [searchResult, setSearchResult] = useState<{ identity_id: string | null; name: string | null; similarity: number } | null>(null);
  const queryClient = useQueryClient();
  const screens = Grid.useBreakpoint();
  const isMobile = !screens.md; // < 768px 视为手机

  const { data, isLoading } = useQuery({
    queryKey: ['faces', page, search],
    queryFn: () => fetchFaces(page, search),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteFace,
    onSuccess: () => {
      message.success('已删除');
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    },
  });

  const batchMutation = useMutation({
    mutationFn: (fd: FormData) => batchImportFaces(fd),
    onSuccess: (res) => {
      setBatchResult(res.data.results);
      const ok = res.data.results.filter((r: any) => r.status === 'ok').length;
      message.success(`导入完成:${ok}/${res.data.total} 成功`);
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '批量导入失败'),
  });

  const searchMutation = useMutation({
    mutationFn: (fd: FormData) => searchFace(fd),
    onSuccess: (res) => setSearchResult(res.data),
    onError: (err: any) => message.error(err?.response?.data?.detail || '识别失败'),
  });

  const faces = data?.data.items || [];
  const total = data?.data.total || 0;

  return (
    <div>
      <div style={{ marginBottom: 20, display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: '1 1 200px', minWidth: 0 }}>
          <Input
            placeholder="搜索姓名"
            prefix={<SearchOutlined />}
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(1); }}
            allowClear
            style={{ flex: 1, minWidth: 120 }}
          />
          <Tag color="blue">{total} 人</Tag>
        </div>
        <Space wrap>
          <Button icon={<ScanOutlined />} onClick={() => { setSearchOpen(true); setSearchResult(null); setSearchFile(null); }} size={isMobile ? 'middle' : 'large'}>
            图片识别
          </Button>
          <Button icon={<UploadOutlined />} onClick={() => { setBatchOpen(true); setBatchResult(null); setBatchFiles([]); setBatchName(''); }} size={isMobile ? 'middle' : 'large'}>
            批量导入
          </Button>
          <Button icon={<VideoCameraOutlined />} onClick={() => setCameraRegisterOpen(true)} size={isMobile ? 'middle' : 'large'}>
            摄像头注册
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setRegisterOpen(true)} size={isMobile ? 'middle' : 'large'}>
            注册新人脸
          </Button>
        </Space>
      </div>

      {faces.length === 0 && !isLoading ? (
        <Card variant="borderless" style={{ borderRadius: 8, textAlign: 'center', padding: 60 }}>
          <UserOutlined style={{ fontSize: 48, color: '#ccc' }} />
          <div style={{ color: '#999', marginTop: 16, fontSize: 15 }}>人脸库为空,点击上方按钮注册</div>
        </Card>
      ) : (
        <Row gutter={[16, 16]}>
          {faces.map((face, i) => (
            <Col key={face.id} xs={24} sm={12} lg={8} xl={6}>
              <Card
                variant="borderless"
                hoverable
                style={{ borderRadius: 8 }}
                actions={[
                  <Button key="edit" type="text" icon={<EditOutlined />} onClick={() => setEditFace(face)} />,
                  <Button
                    key="delete"
                    type="text"
                    danger
                    icon={<DeleteOutlined />}
                    onClick={() => {
                      Modal.confirm({
                        title: '确认删除',
                        content: `确定删除 "${face.name}" 吗?该操作不可撤销。`,
                        okButtonProps: { danger: true },
                        onOk: () => deleteMutation.mutate(face.id),
                      });
                    }}
                  />,
                ]}
              >
                <div style={{ textAlign: 'center' }}>
                  <Avatar
                    size={64}
                    icon={<UserOutlined />}
                    src={face.avatar_path || undefined}
                    style={{ backgroundColor: colors[i % colors.length], marginBottom: 12 }}
                  />
                  <div style={{ fontWeight: 600, fontSize: 16, marginBottom: 4 }}>{face.name}</div>
                  <div style={{ color: '#888', fontSize: 13, marginBottom: 8 }}>
                    注册于 {new Date(face.created_at).toLocaleDateString()}
                  </div>
                  <Space>
                    <Tag color="blue">{face.embedding_count} 个特征</Tag>
                    {face.notes && <Tag>{face.notes}</Tag>}
                  </Space>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      )}

      <FaceRegisterModal open={registerOpen} onClose={() => setRegisterOpen(false)} />
      <CameraRegisterModal open={cameraRegisterOpen} identities={faces} onClose={() => setCameraRegisterOpen(false)} />
      <FaceEditModal face={editFace} open={!!editFace} onClose={() => setEditFace(null)} />

      {/* 批量导入 Modal */}
      <Modal
        title="批量导入人脸"
        open={batchOpen}
        onCancel={() => setBatchOpen(false)}
        footer={batchResult ? (
          <Button type="primary" onClick={() => setBatchOpen(false)}>关闭</Button>
        ) : (
          <Space>
            <Button onClick={() => setBatchOpen(false)}>取消</Button>
            <Button
              type="primary"
              loading={batchMutation.isPending}
              disabled={!batchName || batchFiles.length === 0}
              onClick={() => {
                const fd = new FormData();
                fd.append('name', batchName);
                batchFiles.forEach((f) => fd.append('images', f.originFileObj));
                batchMutation.mutate(fd);
              }}
            >
              开始导入
            </Button>
          </Space>
        )}
        width={520}
      >
        {batchResult ? (
          <div>
            <Descriptions column={2} size="small" style={{ marginBottom: 16 }}>
              <Descriptions.Item label="姓名">{batchName}</Descriptions.Item>
              <Descriptions.Item label="总数">{batchResult.length}</Descriptions.Item>
            </Descriptions>
            {batchResult.map((r, i) => (
              <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid #f0f0f0' }}>
                <span style={{ fontSize: 13 }}>{r.file}</span>
                {r.status === 'ok' ? (
                  <Tag color="green">成功</Tag>
                ) : (
                  <Tag color="red">{r.reason}</Tag>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div>
            <Input placeholder="输入姓名" value={batchName} onChange={(e) => setBatchName(e.target.value)} style={{ marginBottom: 12 }} />
            <Upload
              listType="picture"
              multiple
              fileList={batchFiles}
              beforeUpload={(file) => {
                setBatchFiles((prev) => [...prev, { uid: String(Date.now() + Math.random()), name: file.name, status: 'done', originFileObj: file }]);
                return false;
              }}
              onRemove={(file) => setBatchFiles((prev) => prev.filter((f) => f.uid !== file.uid))}
            >
              <Button icon={<UploadOutlined />}>选择多张照片</Button>
            </Upload>
            <div style={{ color: '#888', fontSize: 12, marginTop: 8 }}>
              上传同一人的多张不同角度照片,系统会逐张检测并提取特征
            </div>
          </div>
        )}
      </Modal>

      {/* 图片识别搜索 Modal */}
      <Modal
        title="图片识别"
        open={searchOpen}
        onCancel={() => setSearchOpen(false)}
        footer={searchResult ? (
          <Button type="primary" onClick={() => setSearchOpen(false)}>关闭</Button>
        ) : (
          <Space>
            <Button onClick={() => setSearchOpen(false)}>取消</Button>
            <Button
              type="primary"
              loading={searchMutation.isPending}
              disabled={!searchFile}
              onClick={() => {
                const fd = new FormData();
                fd.append('image', searchFile.originFileObj);
                searchMutation.mutate(fd);
              }}
            >
              开始识别
            </Button>
          </Space>
        )}
      >
        {searchResult ? (
          <Result
            status={searchResult.identity_id ? 'success' : 'info'}
            title={searchResult.identity_id ? `匹配到: ${searchResult.name}` : '未匹配到已知人员'}
            subTitle={`相似度: ${(searchResult.similarity * 100).toFixed(1)}%`}
          />
        ) : (
          <Upload
            listType="picture"
            maxCount={1}
            fileList={searchFile ? [searchFile] : []}
            beforeUpload={(file) => {
              setSearchFile({ uid: '-1', name: file.name, status: 'done', originFileObj: file });
              return false;
            }}
            onRemove={() => setSearchFile(null)}
          >
            <Button icon={<UploadOutlined />}>选择照片</Button>
          </Upload>
        )}
      </Modal>
    </div>
  );
}
