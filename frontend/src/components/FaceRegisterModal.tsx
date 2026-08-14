import { useState } from 'react';
import { Modal, Form, Input, Upload, Button, message } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { createFace } from '../api/faces';

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function FaceRegisterModal({ open, onClose }: Props) {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState<any[]>([]);
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (formData: FormData) => createFace(formData),
    onSuccess: () => {
      message.success('注册成功');
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      form.resetFields();
      setFileList([]);
      onClose();
    },
    onError: (err: any) => {
      message.error(err.response?.data?.detail || '注册失败');
    },
  });

  const handleOk = () => {
    form.validateFields().then((values) => {
      const fd = new FormData();
      fd.append('name', values.name);
      fd.append('notes', values.notes || '');
      if (fileList[0]?.originFileObj) {
        fd.append('image', fileList[0].originFileObj);
      }
      mutation.mutate(fd);
    });
  };

  return (
    <Modal
      title="注册新人脸"
      open={open}
      onOk={handleOk}
      onCancel={onClose}
      confirmLoading={mutation.isPending}
    >
      <Form form={form} layout="vertical">
        <Form.Item name="name" label="姓名" rules={[{ required: true, message: '请输入姓名' }]}>
          <Input placeholder="输入姓名" />
        </Form.Item>
        <Form.Item name="notes" label="备注">
          <Input.TextArea rows={2} />
        </Form.Item>
        <Form.Item label="照片" required>
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
            <Button icon={<UploadOutlined />}>选择照片</Button>
          </Upload>
        </Form.Item>
      </Form>
    </Modal>
  );
}
