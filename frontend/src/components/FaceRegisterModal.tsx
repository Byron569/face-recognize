import { useEffect, useRef, useState } from 'react';
import { Modal, Form, Input, Upload, Button, message, Spin } from 'antd';
import { UploadOutlined } from '@ant-design/icons';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { batchImportFaces, detectFace } from '../api/faces';

interface Props {
  open: boolean;
  onClose: () => void;
}

interface FaceMark {
  width: number;
  height: number;
  faces: Array<{ bbox: number[]; det_score: number }>;
}

type DetStatus = 'idle' | 'loading' | 'ok' | 'none' | 'error';

export default function FaceRegisterModal({ open, onClose }: Props) {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState<any[]>([]);
  const [previewUid, setPreviewUid] = useState<string | null>(null);
  const [marks, setMarks] = useState<Record<string, FaceMark>>({});
  const [detStatus, setDetStatus] = useState<Record<string, DetStatus>>({});
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const urlRef = useRef<string | null>(null);

  // 清理预览 URL
  useEffect(() => {
    return () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);

  // 切换预览图:生成 URL + 无缓存时触发检测
  useEffect(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setPreviewUrl(null);
    const file = fileList.find((f) => f.uid === previewUid);
    if (file?.originFileObj) {
      const url = URL.createObjectURL(file.originFileObj);
      urlRef.current = url;
      setPreviewUrl(url);
      if (!marks[file.uid] && detStatus[file.uid] !== 'loading') {
        runDetect(file.uid, file.originFileObj);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewUid, fileList]);

  const runDetect = async (uid: string, originFileObj: File) => {
    setDetStatus((prev) => ({ ...prev, [uid]: 'loading' }));
    const fd = new FormData();
    fd.append('image', originFileObj);
    try {
      const res = await detectFace(fd);
      setMarks((prev) => ({
        ...prev,
        [uid]: { width: res.data.width, height: res.data.height, faces: res.data.faces },
      }));
      setDetStatus((prev) => ({ ...prev, [uid]: res.data.faces.length ? 'ok' : 'none' }));
    } catch {
      setDetStatus((prev) => ({ ...prev, [uid]: 'error' }));
    }
  };

  const mutation = useMutation({
    mutationFn: (formData: FormData) => batchImportFaces(formData),
    onSuccess: (res) => {
      const results = res.data?.results ?? [];
      const failed = results.filter((r) => r.status === 'error');
      if (results.length > 0 && failed.length === results.length) {
        // 全部失败:不关闭,提示原因
        message.error(`注册失败:${failed[0].reason || '图片质量不达标'}`);
        return;
      }
      if (failed.length > 0) {
        message.warning(
          `${results.length - failed.length} 张成功,${failed.length} 张失败:${failed[0].reason || '质量不达标'}`
        );
      } else {
        message.success(`注册成功(${results.length} 张照片)`);
      }
      queryClient.invalidateQueries({ queryKey: ['faces'] });
      form.resetFields();
      setFileList([]);
      setPreviewUid(null);
      setMarks({});
      setDetStatus({});
      onClose();
    },
    onError: (err: any) => {
      message.error(err.response?.data?.detail || '注册失败');
    },
  });

  const handleOk = () => {
    if (fileList.length === 0) {
      message.error('请至少选择一张照片');
      return;
    }
    form.validateFields().then((values) => {
      const fd = new FormData();
      fd.append('name', values.name);
      fd.append('notes', values.notes || '');
      fileList.forEach((f) => {
        if (f.originFileObj) {
          fd.append('images', f.originFileObj);
        }
      });
      mutation.mutate(fd);
    });
  };

  const curMark = previewUid ? marks[previewUid] : undefined;
  const curStatus = previewUid ? detStatus[previewUid] : 'idle';

  return (
    <Modal
      title="注册新人脸"
      open={open}
      onOk={handleOk}
      onCancel={onClose}
      confirmLoading={mutation.isPending}
      width={520}
    >
      <Form form={form} layout="vertical">
        <Form.Item name="name" label="姓名" rules={[{ required: true, message: '请输入姓名' }]}>
          <Input placeholder="输入姓名" />
        </Form.Item>
        <Form.Item name="notes" label="备注">
          <Input.TextArea rows={2} />
        </Form.Item>
        <Form.Item label="照片(可多选,建议 3~5 张不同角度)" required>
          <Upload
            listType="picture"
            multiple
            fileList={fileList}
            beforeUpload={(file) => {
              const uid = file.uid;
              setFileList((prev) => [
                ...prev,
                { uid, name: file.name, status: 'done', originFileObj: file },
              ]);
              setPreviewUid(uid);
              return false;
            }}
            onRemove={(file) => {
              setFileList((prev) => prev.filter((f) => f.uid !== file.uid));
              setMarks((prev) => {
                const next = { ...prev };
                delete next[file.uid];
                return next;
              });
              setDetStatus((prev) => {
                const next = { ...prev };
                delete next[file.uid];
                return next;
              });
              if (previewUid === file.uid) {
                const rest = fileList.filter((f) => f.uid !== file.uid);
                setPreviewUid(rest[0]?.uid ?? null);
              }
            }}
            onPreview={(file) => setPreviewUid(file.uid)}
          >
            <Button icon={<UploadOutlined />}>选择照片</Button>
          </Upload>

          {/* ── 大图预览 + 检测框 ── */}
          {previewUrl && (
            <div style={{ marginTop: 12 }}>
              <div
                style={{
                  position: 'relative',
                  display: 'inline-block',
                  maxWidth: '100%',
                  border: '1px solid #e5e5e5',
                  borderRadius: 6,
                  overflow: 'hidden',
                }}
              >
                <img src={previewUrl} alt="预览" style={{ maxWidth: '100%', display: 'block' }} />
                {curMark?.faces.map((f, i) => {
                  const [x1, y1, x2, y2] = f.bbox;
                  return (
                    <div
                      key={i}
                      style={{
                        position: 'absolute',
                        left: `${(x1 / curMark.width) * 100}%`,
                        top: `${(y1 / curMark.height) * 100}%`,
                        width: `${((x2 - x1) / curMark.width) * 100}%`,
                        height: `${((y2 - y1) / curMark.height) * 100}%`,
                        border: '2px solid #52c41a',
                        boxSizing: 'border-box',
                        borderRadius: 4,
                        pointerEvents: 'none',
                      }}
                    />
                  );
                })}
              </div>
              <div style={{ marginTop: 6, fontSize: 13 }}>
                {curStatus === 'loading' && (
                  <span style={{ color: '#888' }}>
                    <Spin size="small" /> 检测中…
                  </span>
                )}
                {curStatus === 'ok' && (
                  <span style={{ color: '#52c41a' }}>
                    检测到 {curMark!.faces.length} 张人脸,该照片可用于注册
                  </span>
                )}
                {curStatus === 'none' && (
                  <span style={{ color: '#faad14' }}>
                    未检测到人脸,注册时该照片会被拒绝,请更换
                  </span>
                )}
                {curStatus === 'error' && (
                  <span style={{ color: '#ff4d4f' }}>检测失败,请重试或更换照片</span>
                )}
              </div>
            </div>
          )}
          <div style={{ color: '#999', fontSize: 12, marginTop: 4 }}>
            点击缩略图可切换预览;多角度照片注册后实时识别更稳
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
}
