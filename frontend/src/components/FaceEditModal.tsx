import { useState } from 'react';
import {
  Modal, Descriptions, Form, Input, Upload, Button, message, Divider,
  Popconfirm, Space, Tag, Badge, Avatar,
} from 'antd';
import {
  UploadOutlined, DeleteOutlined, EditOutlined, UserOutlined, CameraOutlined,
} from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import dayjs from 'dayjs';
import { Identity, updateFace, addFaceEmbedding, deleteFace, uploadAvatar } from '../api/faces';

interface Props {
  face: Identity | null;
  open: boolean;
  onClose: () => void;
}

export default function FaceEditModal({ face, open, onClose }: Props) {
  const [editing, setEditing] = useState(false);
  const [fileList, setFileList] = useState<any[]>([]);
  const [form] = Form.useForm();
  const queryClient = useQueryClient();

  const updateMutation = useMutation({
    mutationFn: (data: Record<string, unknown>) => updateFace(face!.id, data),
    onSuccess: () => {
      message.success('信息已更新');
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      setEditing(false);
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '更新失败'),
  });

  const embedMutation = useMutation({
    mutationFn: (fd: FormData) => addFaceEmbedding(face!.id, fd),
    onSuccess: () => {
      message.success('图片已添加，特征数+1');
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      setFileList([]);
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '添加失败'),
  });

  const deleteMutation = useMutation({
    mutationFn: () => deleteFace(face!.id),
    onSuccess: () => {
      message.success('已删除');
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      onClose();
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '删除失败'),
  });

  const avatarMutation = useMutation({
    mutationFn: (fd: FormData) => uploadAvatar(face!.id, fd),
    onSuccess: () => {
      message.success('头像已更新');
      queryClient.invalidateQueries({ queryKey: ['faces'] });
    },
    onError: (err: any) => message.error(err?.response?.data?.detail || '头像更新失败'),
  });

  if (!face) return null;

  return (
    <Modal
      title={
        <Space>
          <UserOutlined />
          {editing ? (
            <Input
              size="small"
              defaultValue={face.name}
              style={{ width: 160 }}
              onPressEnter={(e) => {
                updateMutation.mutate({ name: (e.target as HTMLInputElement).value });
              }}
            />
          ) : (
            <span>{face.name}</span>
          )}
          <Tag color="blue">{face.embedding_count} 个特征</Tag>
        </Space>
      }
      open={open}
      onCancel={() => { setEditing(false); onClose(); }}
      footer={
        <Space>
          {editing ? (
            <>
              <Button onClick={() => setEditing(false)}>取消</Button>
              <Button
                type="primary"
                loading={updateMutation.isPending}
                onClick={() => {
                  form.validateFields().then((values) => {
                    updateMutation.mutate(values);
                  });
                }}
              >
                保存
              </Button>
            </>
          ) : (
            <>
              <Button icon={<EditOutlined />} onClick={() => {
                form.setFieldsValue({ name: face.name, notes: face.notes });
                setEditing(true);
              }}>
                编辑
              </Button>
              <Popconfirm
                title="确定删除？"
                description="该操作不可撤销，所有特征数据将被清除。"
                onConfirm={() => deleteMutation.mutate()}
                okButtonProps={{ danger: true }}
              >
                <Button danger icon={<DeleteOutlined />} loading={deleteMutation.isPending}>
                  删除
                </Button>
              </Popconfirm>
            </>
          )}
        </Space>
      }
      width={560}
    >
      {editing ? (
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item name="name" label="姓名" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="notes" label="备注">
            <Input.TextArea rows={2} placeholder="可选备注信息" />
          </Form.Item>
        </Form>
      ) : (
        <Descriptions column={2} size="small" bordered style={{ marginTop: 8 }}>
          <Descriptions.Item label="ID" span={2}>
            <code style={{ fontSize: 12 }}>{face.id}</code>
          </Descriptions.Item>
          <Descriptions.Item label="姓名">{face.name}</Descriptions.Item>
          <Descriptions.Item label="特征数">
            <Badge count={face.embedding_count} color="blue" />
          </Descriptions.Item>
          <Descriptions.Item label="注册时间" span={2}>
            {dayjs(face.created_at).format('YYYY-MM-DD HH:mm:ss')}
          </Descriptions.Item>
          <Descriptions.Item label="备注" span={2}>
            {face.notes || <span style={{ color: '#bbb' }}>无</span>}
          </Descriptions.Item>
        </Descriptions>
      )}

      <Divider style={{ margin: '16px 0 12px' }} />
      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
        <Avatar
          size={64}
          icon={<UserOutlined />}
          src={face.avatar_path || undefined}
          style={{ backgroundColor: '#1677ff' }}
        />
        <div>
          <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>头像</div>
          <Upload
            showUploadList={false}
            beforeUpload={(file) => {
              const fd = new FormData();
              fd.append('image', file);
              avatarMutation.mutate(fd);
              return false;
            }}
          >
            <Button icon={<CameraOutlined />} loading={avatarMutation.isPending} size="small">
              更换头像
            </Button>
          </Upload>
          <div style={{ color: '#888', fontSize: 12, marginTop: 4 }}>仅更新展示图，不重新提取特征</div>
        </div>
      </div>

      <Divider style={{ margin: '8px 0 12px' }} />
      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>追加特征图片</div>
      <div style={{ color: '#888', fontSize: 12, marginBottom: 8 }}>
        上传同一人的不同角度照片，可提高识别准确率
      </div>
      <Upload
        listType="picture"
        maxCount={1}
        fileList={fileList}
        beforeUpload={(file) => {
          setFileList([{ uid: '-1', name: file.name, status: 'done', originFileObj: file }]);
          return false;
        }}
        onRemove={() => setFileList([])}
      >
        <Button
          icon={<UploadOutlined />}
          loading={embedMutation.isPending}
          onClick={() => {
            if (fileList.length === 0) return;
            const fd = new FormData();
            fd.append('image', fileList[0].originFileObj);
            embedMutation.mutate(fd);
          }}
        >
          选择照片
        </Button>
      </Upload>
      {fileList.length > 0 && (
        <Button
          type="primary"
          size="small"
          style={{ marginTop: 8 }}
          loading={embedMutation.isPending}
          onClick={() => {
            const fd = new FormData();
            fd.append('image', fileList[0].originFileObj);
            embedMutation.mutate(fd);
          }}
        >
          确认上传
        </Button>
      )}
    </Modal>
  );
}
