#include <cstdio>
#include <net.h>
#include "gfpgan.h"
#include "face.h"
#include "realesrgan.h"

#define RESTORE_WHOLE_IMAGE 1   //0-only restore face, 1-restore whole image
#define RESTORE_IMAGE_COLOR 0   //0-no color image, 1-coloring grayscale images

static void to_ocv(const ncnn::Mat &result, cv::Mat &out) {
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(512, 512), CV_32FC3);
    for (int i = 0; i < result.h; i++) {
        for (int j = 0; j < result.w; j++) {
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = (result.channel(0)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = (result.channel(1)[i * result.w + j] + 1) / 2;
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = (result.channel(2)[i * result.w + j] + 1) / 2;
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);

    cv_result_8U.copyTo(out);

}

#if RESTORE_WHOLE_IMAGE

static void paste_faces_to_input_image(const cv::Mat &restored_face, cv::Mat &trans_matrix_inv, cv::Mat &bg_upsample) {
    trans_matrix_inv.at<float>(0, 2) += 1.0;
    trans_matrix_inv.at<float>(1, 2) += 1.0;

    cv::Mat inv_restored;
    cv::warpAffine(restored_face, inv_restored, trans_matrix_inv, bg_upsample.size(), 1, 0);
    cv::Mat mask = cv::Mat::ones(cv::Size(512, 512), CV_8UC1) * 255;

    cv::Mat inv_mask;
    cv::warpAffine(mask, inv_mask, trans_matrix_inv, bg_upsample.size(), 1, 0);

    cv::Mat inv_mask_erosion;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
    cv::erode(inv_mask, inv_mask_erosion, kernel);

    cv::Mat pasted_face;
    cv::bitwise_and(inv_restored, inv_restored, pasted_face, inv_mask_erosion);

    int total_face_area = cv::countNonZero(inv_mask_erosion);
    int w_edge = int(std::sqrt(total_face_area) / 20);
    int erosion_radius = w_edge * 2;
    cv::Mat inv_mask_center;

    kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(erosion_radius, erosion_radius));
    cv::erode(inv_mask_erosion, inv_mask_center, kernel);

    int blur_size = w_edge * 2;
    cv::Mat inv_soft_mask;
    cv::GaussianBlur(inv_mask_center, inv_soft_mask, cv::Size(blur_size + 1, blur_size + 1), 0, 0, 4);

    for (int h = 0; h < bg_upsample.rows; h++) {
        for (int w = 0; w < bg_upsample.cols; w++) {
            float alpha = inv_soft_mask.at<uchar>(h, w) / 255.0;
            bg_upsample.at<cv::Vec3b>(h, w)[0] =
                    pasted_face.at<cv::Vec3b>(h, w)[0] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[0];
            bg_upsample.at<cv::Vec3b>(h, w)[1] =
                    pasted_face.at<cv::Vec3b>(h, w)[1] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[1];
            bg_upsample.at<cv::Vec3b>(h, w)[2] =
                    pasted_face.at<cv::Vec3b>(h, w)[2] * alpha + (1 - alpha) * bg_upsample.at<cv::Vec3b>(h, w)[2];
        }
    }
}

#endif

#include <gpu.h>
#include <string>
#include <cstring>
#include <cstdlib>

int main(int argc, char **argv) {
    std::string inputpath;
    std::string outputpath = "result.png";
    std::string modelpath = "models";
    int gpu_id = ncnn::get_default_gpu_index();

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            inputpath = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outputpath = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            modelpath = argv[++i];
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            gpu_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stderr, "Usage: %s -i infile -o outfile [options]...\n\n", argv[0]);
            fprintf(stderr, "  -h                   show this help\n");
            fprintf(stderr, "  -i input-path        input image path (jpg/png/webp)\n");
            fprintf(stderr, "  -o output-path       output image path (jpg/png/webp) (default: result.png)\n");
            fprintf(stderr, "  -m model-path        folder path to the pre-trained models (default: models)\n");
            fprintf(stderr, "  -g gpu-id            gpu device to use (default: auto)\n");
            return 0;
        }
    }

    if (inputpath.empty()) {
        fprintf(stderr, "Usage: %s -i infile -o outfile [options]...\n", argv[0]);
        return -1;
    }

    cv::Mat img = cv::imread(inputpath, 1);
    if (img.empty()) {
        fprintf(stderr, "cv::imread %s failed\n", inputpath.c_str());
        return -1;
    }

    GFPGAN gfpgan;
    gfpgan.load(modelpath + "/encoder.param", modelpath + "/encoder.bin", modelpath + "/style.bin");

#if RESTORE_WHOLE_IMAGE
    Face face_detector;
    face_detector.load(modelpath + "/yolov5-blazeface.param", modelpath + "/yolov5-blazeface.bin");

    RealESRGAN real_esrgan(gpu_id);
    real_esrgan.load(modelpath + "/real_esrgan.param", modelpath + "/real_esrgan.bin");

    cv::Mat bg_upsample;
    real_esrgan.tile_process(img, bg_upsample);

    std::vector<cv::Mat> trans_img;
    std::vector<cv::Mat> trans_matrix_inv;
    std::vector<Object> objects;
    face_detector.detect(img, objects);
    face_detector.align_warp_face(img, objects, trans_matrix_inv, trans_img);

    for (size_t i = 0; i < objects.size(); i++) {
        ncnn::Mat gfpgan_result;
        gfpgan.process(trans_img[i], gfpgan_result);

        cv::Mat restored_face;
        to_ocv(gfpgan_result, restored_face);

        paste_faces_to_input_image(restored_face, trans_matrix_inv[i], bg_upsample);

    }
    cv::imwrite(outputpath, bg_upsample);
#else
    ncnn::Mat gfpgan_result;
    gfpgan.process(img, gfpgan_result);

    cv::Mat restored_face;
    to_ocv(gfpgan_result, restored_face);
    cv::imwrite(outputpath, restored_face);
#endif

    return 0;
}
