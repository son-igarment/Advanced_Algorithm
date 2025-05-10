# -*- coding: utf-8 -*-
"""
Unicode Algorithm for Vietnamese Address Standardization

Author: Phạm Lê Ngọc Sơn

This algorithm standardizes Vietnamese addresses by normalizing text,
correcting common character errors, and matching against standard databases.

Original file is located at
    https://colab.research.google.com/drive/1EXZXx5d07B4LWRFz9aPR_GFrVHElLGT2
"""

# NOTE: you CAN change this cell
# If you want to use your own database, download it here
# !gdown ...
!gdown --fuzzy https://drive.google.com/file/d/1oKYrDb7uW-XSYUBbZTK7UAASCf9SpDMl/view?usp=sharing -O list_province_standard.txt

!gdown --fuzzy https://drive.google.com/file/d/1avKxiUxvtNwvG7_yW___QjZJ2Kvk_vDh/view?usp=sharing -O list_district_standard.txt

!gdown --fuzzy https://drive.google.com/file/d/1vPknKB7iX7-Ziag8nuPGOVdB3w9icmkb/view?usp=sharing -O list_ward_standard.txt

!gdown --fuzzy https://drive.google.com/file/d/1pMdpxHE6ta_NaRXJKe_Mjn8REJKSmEsV/view?usp=sharing -O province_abbreviations.json

# NOTE: you CAN change this cell
# Add more to your needs
# you must place ALL pip install here
!pip install editdistance

# NOTE: you CAN change this cell
# import your library here
import time
import re
import json
import unicodedata
import numpy as np

# NOTE: you MUST change this cell
# New methods / functions must be written under class Solution.
class Solution:
    def __init__(self):
        # list provice, district, ward for private test, do not change for any reason (these file will be provided later with this exact name)
        self.province_path = 'list_province.txt'
        self.district_path = 'list_district.txt'
        self.ward_path = 'list_ward.txt'

        # Đường dẫn chuẩn hóa
        self.province_standard_path = 'list_province_standard.txt'
        self.district_standard_path = 'list_district_standard.txt'
        self.ward_standard_path = 'list_ward_standard.txt'

        # Danh sách các pattern để normalize text
        self.patterns = [
            (re.compile(r',[a-zA-Z]{1,2}\.'), ' '),
            (re.compile(r'(?<=\w)-(?=\w)'), ' '),
            (re.compile(r'^[A-Za-z]?\d+\/\d*[A-Za-z]*,?'), ' '),
            (re.compile(
                r'(?:[qQ][uU][ậẬ][nN]\s?|[hH][uU][yY][ệỆ][nN]\s?|[pP][hH][ưƯ][ờỜ][nN][gG]\s?|[xX][ãÃ]\s?|[tT][hH][ịỊ]\s?[xX][ãÃ]\s?\b|[tT][hH][àÀ][nN][hH]\s?[pP][hH][ốỐ]\s?|[tT][hH][ịỊ]\s?[tT][rR][ấẤ][nN]|[tT][hH][ỉỈ][nN][hH]\s?)\s?'),
             ' '),
            (re.compile(
                r'(?:^|(?<=[\s,]))(?:[qQ][uU][ậẬ][nN]\s?|[hH][uU][yY][ệỆ][nN]\s?|[pP][hH][ưƯ][ờỜ][nN][gG]\s?|[xX][ãÃ]\s?|[tT][hH][ịỊ]\s?[xX][ãÃ]\s?\b|[tT][hH][àÀ][nN][hH]\s?[pP][hH][ốỐ]\s?|[tT][hH][ịỊ]\s?[tT][rR][ấẤ][nN]|[tT][hH][àÀ][nN][hH]\s?[pP][hH][ốỐ]|[tT][ỉỈ][nN][hH]\s?)\s?'),
             ' '),
            (re.compile(
                r'(?:^|(?<=[\s,]))([Tt]\.|[Tt]\s|[Tt][Pp]\.|[Tt][Pp]\s|[Qq]\.|[Qq](?=\.|\d)|[Hh]\.|[Hh]\s|[Tt][Xx]\.|[Tt][Xx]\s|[Pp]\.|[Pp](?=\.|\d)|[Xx]\.|[Xx]\s|[Tt][Tt]\.|[Tt][Tt]\s|[fF]\.|[fF]\s)'),
             ' '),
            (re.compile(
                r'^(?:[sS]ố\s[nN]hà\s\d+|[sS]ố\s\d+|[kK]hu\s(?:[pP]hố\s)?\d+|'
                r'[tT]ổ\s(?:[dD]ân\s[pP]hố\s)?\d+)(?:,\s[tT]ổ\s\d+|,\s[kK]hu\s(?:[pP]hố\s)?\d+)?'),
             ' '),
            (re.compile(r'[.,]'), ' '),
            (re.compile(r'\s+'), ' ')
        ]
        self.special = ['+', '-', '*', '/', '_', '?', '<', '>', '=', ':', ';', '.', ',']

        # =============================================================================
        # MA TRẬN TRỌNG SỐ ĐÃ SỬA LỖI UNICODE
        # =============================================================================
        self.VIETNAMESE_CHARS = [
            'a', 'à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ',
            'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ', 'e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ',
            'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ', 'i', 'ì', 'í', 'ỉ', 'ĩ', 'ị',
            'o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ',
            'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ', 'u', 'ù', 'ú', 'ủ', 'ũ', 'ụ',
            'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự', 'y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'
        ]

        self.ABBREVIATIONS = self.load_abbreviations("province_abbreviations.json")
        self.CORRECTED_VIETNAMESE_CHARS = {
            "ià": "ìa", "iá": "ía", "iả": "ỉa", "iã": "ĩa", "iạ": "ịa",
            "uà": "ùa", "uá": "úa", "uả": "ủa", "uã": "ũa", "uạ": "ụa",
            "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
            "oì": "òi", "oí": "ói", "oỉ": "ỏi", "oĩ": "õi", "oị": "ọi",
            "ưà": "ừa", "ưá": "ứa", "ưả": "ửa", "ưã": "ữa", "ưạ": "ựa",
            "aì": "ài", "aí": "ái", "aỉ": "ải", "aĩ": "ãi", "aị": "ại",
            "aò": "ào", "aó": "áo", "aỏ": "ảo", "aõ": "ão", "aọ": "ạo",
            "aù": "àu", "aú": "áu", "aủ": "ảu", "aũ": "ãu", "aụ": "ạu",
            "eò": "èo", "eó": "éo", "eỏ": "ẻo", "eõ": "ẽo", "eọ": "ẹo",
            "âù": "ầu", "âú": "ấu", "âủ": "ẩu", "âũ": "ẫu", "âụ": "ậu",
            "êù": "ều", "êú": "ếu", "êủ": "ểu", "êũ": "ễu", "êụ": "ệu"
        }

        self.SUB_MATRIX = self.build_substitution_matrix()

        self.GROUPS_DISTRICT = {
            'Hoà Bình': 'Hòa Bình',
            'Kbang': 'KBang',
            'Qui Nhơn': 'Quy Nhơn'
        }
        self.GROUPS_WARD = {
            '01': '1',
            '02': '2',
            '03': '3',
            '04': '4',
            '05': '5',
            '06': '6',
            '07': '7',
            '08': '8',
            '09': '9',
            'ái Nghĩa': 'Ái Nghĩa',
            'ái Quốc': 'Ái Quốc',
            'ái Thượng': 'Ái Thượng',
            'ái Tử': 'Ái Tử',
            'ấm Hạ': 'Ấm Hạ',
            'An ấp': 'An Ấp',
            'ẳng Cang': 'Ẳng Cang',
            'ẳng Nưa': 'Ẳng Nưa',
            'ẳng Tở': 'Ẳng Tở',
            'An Hoà': 'An Hòa',
            'Ayun': 'AYun',
            'Bắc ái': 'Bắc Ái',
            'Bảo ái': 'Bảo Ái',
            'Bình Hoà': 'Bình Hòa',
            'Châu ổ': 'Châu Ổ',
            'Chư á': 'Chư Á',
            'Chư Rcăm': 'Chư RCăm',
            'Cộng Hoà': 'Cộng Hòa',
            'Cò  Nòi': 'Cò Nòi',
            'Đại Ân  2': 'Đại Ân 2',
            'Đak ơ': 'Đak Ơ',
            "Đạ M'ri": "Đạ M'Ri",
            'Đông Hoà': 'Đông Hòa',
            'Đồng ích': 'Đồng Ích',
            'Hải Châu  I': 'Hải Châu I',
            'Hải Hoà': 'Hải Hòa',
            'Hành Tín  Đông': 'Hành Tín Đông',
            'Hiệp Hoà': 'Hiệp Hòa',
            'Hoà Bắc': 'Hòa Bắc',
            'Hoà Bình': 'Hòa Bình',
            'Hoà Châu': 'Hòa Châu',
            'Hoà Hải': 'Hòa Hải',
            'Hoà Hiệp Trung': 'Hòa Hiệp Trung',
            'Hoà Liên': 'Hòa Liên',
            'Hoà Lộc': 'Hòa Lộc',
            'Hoà Lợi': 'Hòa Lợi',
            'Hoà Long': 'Hòa Long',
            'Hoà Mạc': 'Hòa Mạc',
            'Hoà Minh': 'Hòa Minh',
            'Hoà Mỹ': 'Hòa Mỹ',
            'Hoà Phát': 'Hòa Phát',
            'Hoà Phong': 'Hòa Phong',
            'Hoà Phú': 'Hòa Phú',
            'Hoà Phước': 'Hòa Phước',
            'Hoà Sơn': 'Hòa Sơn',
            'Hoà Tân': 'Hòa Tân',
            'Hoà Thuận': 'Hòa Thuận',
            'Hoà Tiến': 'Hòa Tiến',
            'Hoà Trạch': 'Hòa Trạch',
            'Hoà Vinh': 'Hòa Vinh',
            'Hương Hoà': 'Hương Hòa',
            'ích Hậu': 'Ích Hậu',
            'ít Ong': 'Ít Ong',
            'Khánh Hoà': 'Khánh Hòa',
            'Krông Á': 'KRông á',
            'Lộc Hoà': 'Lộc Hòa',
            'Minh Hoà': 'Minh Hòa',
            'Mường ải': 'Mường Ải',
            'Mường ẳng': 'Mường Ẳng',
            'Nậm ét': 'Nậm Ét',
            'Nam Hoà': 'Nam Hòa',
            'Na ư': 'Na Ư',
            'Ngã sáu': 'Ngã Sáu',
            'Nghi Hoà': 'Nghi Hòa',
            'Nguyễn Uý': 'Nguyễn Úy',
            'Nguyễn úy': 'Nguyễn Úy',
            'Nhân Hoà': 'Nhân Hòa',
            'Nhơn Hoà': 'Nhơn Hòa',
            'Nhơn nghĩa A': 'Nhơn Nghĩa A',
            'Phúc ứng': 'Phúc Ứng',
            'Phước Hoà': 'Phước Hòa',
            'Sơn Hoá': 'Sơn Hóa',
            'Tạ An Khương  Đông': 'Tạ An Khương Đông',
            'Tạ An Khương  Nam': 'Tạ An Khương Nam',
            'Tăng Hoà': 'Tăng Hòa',
            'Tân Hoà': 'Tân Hòa',
            'Tân Hòa  Thành': 'Tân Hòa Thành',
            'Tân  Khánh Trung': 'Tân Khánh Trung',
            'Tân lợi': 'Tân Lợi',
            'Thái Hoà': 'Thái Hòa',
            'Thiết ống': 'Thiết Ống',
            'Thuận Hoà': 'Thuận Hòa',
            'Thượng ấm': 'Thượng Ấm',
            'Thuỵ Hương': 'Thụy Hương',
            'Thuỷ Xuân': 'Thủy Xuân',
            'Tịnh ấn Đông': 'Tịnh Ấn Đông',
            'Tịnh ấn Tây': 'Tịnh Ấn Tây',
            'Triệu ái': 'Triệu Ái',
            'Triệu ẩu': 'Triệu Ẩu',
            'Trung Hoà': 'Trung Hòa',
            'Trung ý': 'Trung Ý',
            'Tùng ảnh': 'Tùng Ảnh',
            'úc Kỳ': 'Úc Kỳ',
            'ứng Hoè': 'Ứng Hoè',
            'Vĩnh Hoà': 'Vĩnh Hòa',
            'Vũ Hoà': 'Vũ Hòa',
            'Xuân ái': 'Xuân Ái',
            'Xuân áng': 'Xuân Áng',
            'Xuân Hoà': 'Xuân Hòa',
            'Xuất Hoá': 'Xuất Hóa',
            'ỷ La': 'Ỷ La'
        }

        # Tải dữ liệu và xây dựng Trie cho province, district, ward
        self.provinces = list()
        self.province_list = self.load_data(self.province_path, dict())
        # self.province_list = self.load_data_standard(self.province_path)
        self.province_standard_list = self.load_data_standard(self.province_standard_path)
        self.province_node = self.Node()
        self.provinces = self._build_trie(self.province_list, self.province_standard_list, self.province_node, level=1)

        self.districts = list()
        self.district_list = self.load_data(self.district_path, self.GROUPS_DISTRICT)
        # self.district_list = self.load_data_standard(self.district_path)
        self.district_standard_list = self.load_data_standard(self.district_standard_path)
        self.district_node = self.Node()
        self.districts = self._build_trie(self.district_list, self.district_standard_list, self.district_node, level=2)

        self.wards = list()
        self.ward_list = self.load_data(self.ward_path, self.GROUPS_WARD)
        # self.ward_list = self.load_data_standard(self.ward_path)
        self.ward_standard_list = self.load_data_standard(self.ward_standard_path)
        self.ward_node = self.Node()
        self.wards = self._build_trie(self.ward_list, self.ward_standard_list, self.ward_node, level=3)

        # Các biến trạng thái xử lý địa chỉ
        self.address = ""
        self.address_arr = []
        self.province = None
        self.district = None
        self.ward = ""
        self.start = self.end = 0
        self.finish = False
        self.word_count = 0
        self.has_districts = True

    # -------------------------------------------------------------------------
    # Lớp Node cho Trie
    class Node:
        def __init__(self):
            self.word = None  # Từ mà node đại diện
            self.wards = list()  # Thông tin ward (nếu có)
            self.districts = list()  # Thông tin district (nếu có)
            self.provinces = list()  # Thông tin province (nếu có)
            self.children = {}  # Các node con, key là chữ thường của từ
            self.level = 0  # Mức độ của node trong trie
            self.is_terminal = False  # Đánh dấu kết thúc một từ

    # -------------------------------------------------------------------------
    def normalize_text(self, text: str) -> str:
        """Chuẩn hóa chuỗi theo các regex đã định nghĩa."""
        for pattern, replacement in self.patterns:
            text = pattern.sub(replacement, text)
        return text.strip()

    def load_abbreviations(self, filename: str) -> dict:
        with open(filename, encoding="utf8") as f:
            return json.load(f)

    def load_data(self, filename: str, groups: dict) -> list:
        """Đọc file và trả về danh sách các dòng."""
        with open(filename, encoding="utf8") as f:
            lines = [line.strip() for line in f]
            titles = []
            if len(groups) > 0:
                for index, word in enumerate(lines):
                    if not groups.get(word.strip()):
                        titles.append(word.strip())
                    elif groups.get(word.strip()) and lines[index + 1].strip() != groups.get(
                            word.strip()) and not word.strip().isnumeric():
                        titles.append(word.strip())
            else:
                return lines
            return titles

    def load_data_standard(self, filename: str) -> list:
        """Đọc file và trả về danh sách các dòng."""
        with open(filename, encoding="utf8") as f:
            return [line.strip() for line in f]

    def init_process(self):
        """Khởi tạo lại các biến trạng thái."""
        self.address = ""
        self.address_arr = []
        self.province = None
        self.district = None
        self.ward = ""
        self.start = self.end = 0
        self.finish = False
        self.word_count = 0
        self.has_districts = True

    # -------------------------------------------------------------------------
    def insert_node(self, node: Node, data: str, level: int, province=None, district=None, ward=None) -> None:
        """Chèn dữ liệu vào trie theo thứ tự từ cuối đến đầu."""
        words = data.split()
        for i in range(len(words) - 1, -1, -1):
            word = words[i].strip()
            word_lower = word.lower()
            new_word = self.CORRECTED_VIETNAMESE_CHARS.get(word_lower)
            word_lower = new_word if new_word else word_lower
            if word_lower not in node.children:
                new_node = self.Node()
                new_node.word = word
                new_node.level = level
                node.children[word_lower] = new_node
            node = node.children[word_lower]
            # Ở node cuối cùng, gán thông tin liên quan
            # if i == 0:
            node.provinces.append(province)
            node.districts.append(district)
            node.wards.append(ward)
        node.is_terminal = True

    def _build_trie(self, data_list: list, data_standard_list: list, data_node: Node, level: int):
        """
        Xây dựng trie từ data_list và data_standard_list.
        Nếu data_list chứa nhiều hoặc ít phần tử hơn data_standard_list, cập nhật data_standard_list.
        """
        datas = list()
        if len(data_list) >= len(data_standard_list):
            for data in data_list:
                if level == 1:
                    if data not in data_standard_list:
                        data_standard_list.append(data)
                else:
                    exists = any(
                        data == ds for data_standard in data_standard_list
                        for ds in data_standard.split(' , ')
                    )
                    if not exists:
                        data_standard_list.append(data)
            datas = data_standard_list
        else:
            for data in data_list:
                if level == 1:
                    datas.append(data)
                else:
                    exists = False
                    for data_standard in data_standard_list:
                        ds = data_standard.split(' , ')[0]
                        if data == ds:
                            exists = True
                            datas.append(data_standard)
                            break
                    if not exists:
                        datas.append(data)

        if level == 1:
            for line in datas:
                parts = line.split(' , ')
                if parts[0] in data_list:
                    self.insert_node(data_node, parts[0], level, province=parts[0])

        elif level == 2:
            for line in datas:
                parts = line.split(' , ')
                if parts[0] in data_list:
                    self.insert_node(data_node, parts[0], level,
                                     province=(None if len(parts) == 1 else parts[1]),
                                     district=parts[0])
        elif level == 3:
            for line in datas:
                parts = line.split(' , ')
                if parts[0] in data_list:
                    self.insert_node(data_node, parts[0], level,
                                     province=(None if len(parts) == 1 else parts[2]),
                                     district=(None if len(parts) == 1 else parts[1]),
                                     ward=parts[0])

        return datas

    # -------------------------------------------------------------------------
    def kmp(self, pattern: str, text: str) -> bool:
        """
        Tìm kiếm chuỗi pattern trong text sử dụng thuật toán KMP.
        Cập nhật self.start, self.end nếu tìm thấy.
        """
        pat_len = len(pattern)
        text_len = len(text)
        failure = self.get_failure_array(pattern)

        i = text_len - 1
        j = pat_len - 1
        self.start = self.end = i

        while i >= 0:
            # Nếu cả ký tự của pattern và text đều là số nhưng không khớp thì thoát
            if pattern[j].isnumeric() and text[i].isnumeric() and pattern[j] != text[i]:
                return False

            if pattern[j] == text[i]:
                if j == 0:
                    self.start = i
                    self.end = i + pat_len - 1
                    return True
                j -= 1
            elif j < pat_len - 1:
                j = pat_len - 1 - failure[pat_len - 1 - j]
                continue
            i -= 1
        return False

    def get_failure_array(self, pattern: str) -> list:
        """Xây dựng mảng failure cho thuật toán KMP."""
        pat_len = len(pattern)
        failure = [0] * pat_len
        i = 0
        for j in range(1, pat_len):
            while i > 0 and pattern[i] != pattern[j]:
                i = failure[i - 1]
            if pattern[i] == pattern[j]:
                i += 1
            failure[j] = i
        return failure

    # -------------------------------------------------------------------------
    def search_kmp_trie(self, node: Node, s: str, is_root: bool, is_word: bool) -> str:
        """
        Tìm kiếm bằng thuật toán KMP trên trie.
        Trả về kết quả là từ tìm được hoặc chuỗi rỗng nếu không tìm thấy.
        """
        s_lower = s.lower()
        for word, child in node.children.items():
            if word in s_lower:
                if self.kmp(word, s_lower):
                    if not child.is_terminal:
                        name = self.search_kmp_trie(child, s[:self.start], False, is_word)
                        if self.finish:
                            result = s[self.start:].rstrip()
                            real_result = f"{name} {child.word}" if name else ""
                            if is_root and (real_result not in result and real_result.replace(" ", "") not in result):
                                self.finish = False
                                self.address = s
                                continue
                            elif is_root and (real_result in result or real_result.replace(" ", "") in result):
                                self.address = self.address.replace(s, "")
                            return real_result
                    else:
                        if child.level == 2 and child.provinces != self.province:
                            return ""
                        elif child.level == 3 and (
                                self.district not in child.districts and self.province not in child.provinces):
                            return ""

                        if is_root:
                            result = s[self.start:].rstrip()
                            real_result = word
                            if real_result not in result:
                                self.finish = False
                                self.address = s
                                continue
                        if not is_word:
                            self.address = s[:self.start].rstrip()
                        self.finish = True
                        return child.word
        return ""

    # -------------------------------------------------------------------------
    def build_substitution_matrix(self):
        size = len(self.VIETNAMESE_CHARS)
        matrix = np.ones((size, size))

        vowel_groups = [
            {'a', 'ă', 'â'}, {'e', 'ê'}, {'o', 'ô', 'ơ'}, {'u', 'ư'}, {'i', 'y'}
        ]

        for i, char_i in enumerate(self.VIETNAMESE_CHARS):
            for j, char_j in enumerate(self.VIETNAMESE_CHARS):
                if i == j:
                    matrix[i][j] = 0
                    continue

                # Sửa lỗi: Chuẩn hóa NFD để tách base character và dấu
                base_i = unicodedata.normalize('NFD', char_i)[0]
                base_j = unicodedata.normalize('NFD', char_j)[0]

                if base_i == base_j:
                    matrix[i][j] = 0.3  # Cùng base, khác dấu
                elif any({base_i, base_j}.issubset(g) for g in vowel_groups):
                    matrix[i][j] = 0.6  # Cùng nhóm nguyên âm
                else:
                    matrix[i][j] = 1.0  # Khác hoàn toàn

        return matrix

    # =============================================================================
    # HÀM TÍNH KHOẢNG CÁCH
    # =============================================================================
    def vietnamese_edit_distance(self, str1, str2):
        # Chuẩn hóa và xử lý lowercase
        str1 = unicodedata.normalize('NFC', str1.lower())
        str2 = unicodedata.normalize('NFC', str2.lower())

        m, n = len(str1), len(str2)
        dp = np.zeros((m + 1, n + 1), dtype=float)

        # Khởi tạo ma trận với float
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Tính toán từng ô
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                char1, char2 = str1[i - 1], str2[j - 1]

                # Tra cứu substitution cost
                if char1 in self.VIETNAMESE_CHARS and char2 in self.VIETNAMESE_CHARS:
                    idx1 = self.VIETNAMESE_CHARS.index(char1)
                    idx2 = self.VIETNAMESE_CHARS.index(char2)
                    sub_cost = self.SUB_MATRIX[idx1][idx2]
                else:
                    sub_cost = 0.0 if char1 == char2 else 1.0

                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Xóa
                    dp[i][j - 1] + 1,  # Chèn
                    dp[i - 1][j - 1] + sub_cost  # Thay thế
                )

        return round(dp[m][n], 2)  # Làm tròn 2 số thập phân

    def search_minimum_edit_distance(self, datas: list, address_arr: list, level: int, province: str = None,
                                     district: str = None):
        if not address_arr:
            return "", self.address_arr, self.address

        index_data = 0
        results = list()
        words = ""
        end_loop = 0
        min_dis = 0

        len_address_arr = len(address_arr)
        i = len_address_arr - 1
        while i >= end_loop and i >= 0:
            words = address_arr[i] + " " + words
            words_lower = words.lower().rstrip()

            minimum_distance = float('inf')

            data_temp = list()
            if len(results) <= 0:
                data_temp = datas
            else:
                data_temp = results.copy()

            for index, data in enumerate(data_temp):
                arr = data.split()

                if level == 2:
                    data_split = data.split(",")
                    if len(data_split) < 2 or data_split[1].strip() != province:
                        continue
                    arr = data_split[0].split()
                elif level == 3:
                    data_split = data.split(",")
                    if len(data_split) < 3 or (data_split[2].strip() != province or data_split[1].strip() != district):
                        continue
                    arr = data_split[0].split()

                len_arr = len(arr)
                current_len_address_arr = len(words.split())
                if len_arr >= current_len_address_arr:
                    text = " ".join(arr[len_arr - current_len_address_arr:])
                    text_lower = text.lower()

                    distance = self.vietnamese_edit_distance(text_lower, words_lower)
                    if distance < minimum_distance:
                        results.clear()
                        minimum_distance = distance
                        index_data = index
                        if len(results) == 0:
                            results.append(data)
                        else:
                            results[0] = data
                    elif distance == minimum_distance:
                        minimum_distance = distance
                        index_data = index
                        results.append(data)

            if len(results) == 0:
                return "", self.address_arr, self.address

            if level == 1:
                end_loop = len_address_arr - len(results[0].split())
            elif level == 2:
                end_loop = len_address_arr - len(results[0].replace(" , " + province, "").split())
            elif level == 3:
                end_loop = len_address_arr - len(results[0].replace(" , " + district + " , " + province, "").split())

            min_dis = minimum_distance
            i -= 1

        return_address_arr = address_arr
        return_address = " ".join(return_address_arr)
        return_result = ""

        end_loop = 0 if end_loop < 0 else end_loop
        if len(address_arr[end_loop:]) == 1 and address_arr[end_loop][0].isnumeric():
            num = address_arr[end_loop][0]
            if level == 1:
                if not any(num == result[0] for result in results):
                    return return_result, return_address_arr, return_address
            elif level == 2:
                if not any(num == str(result[0].split(",")[0]).strip() for result in results):
                    return return_result, return_address_arr, return_address
            elif level == 3:
                if not any(num == str(result[0].split(",")[0]).strip() for result in results):
                    return return_result, return_address_arr, return_address

        results.sort(reverse=True)

        if min_dis <= 2 and end_loop >= 0:
            if level == 1:
                return_result = str(results[0])
                return_address_arr = address_arr[:len_address_arr - len(return_result.split())]
                return_address = " ".join(return_address_arr)
            elif level == 2:
                return_result = str(results[0].split(",")[0]).strip()
                return_address_arr = address_arr[:len_address_arr - len(return_result.split())]
                return_address = " ".join(return_address_arr)
            elif level == 3:
                return_result = str(results[0].split(",")[0]).strip()
                return_address_arr = address_arr[:len_address_arr - len(return_result.split())]
                return_address = " ".join(return_address_arr)

        return return_result, return_address_arr, return_address

    # -------------------------------------------------------------------------
    def search_trie(self, datas: list, node: Node, s_arr: list, level: int, is_root: bool) -> str:
        """
        Tìm kiếm trên trie bằng cách duyệt mảng các từ (s_arr).
        Xử lý các trường hợp khi không tìm thấy node con hoặc ký tự đặc biệt.
        """
        if not s_arr:
            return ""

        i = len(s_arr) - 1
        while i >= 0:
            s_lower = s_arr[i].lower()
            self.word_count += 1
            if len(s_lower) == 1 and s_lower not in self.VIETNAMESE_CHARS and not s_lower.isnumeric() and s_lower not in self.special:
                i -= 1
                continue
            child_node = node.children.get(s_lower)

            if child_node:
                remaining_s_arr = s_arr[:i]
                if not child_node.is_terminal:
                    word = self.search_trie(datas, child_node, remaining_s_arr, level, False)
                    if is_root:
                        if child_node.level == 1 and not word:
                            name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level)
                            self.address = address
                            self.address_arr = address_arr
                            return name
                        elif child_node.level == 2 and not word:
                            name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                           province=self.province)
                            if not name:
                                self.address_arr = address_arr[:len(address_arr) - self.word_count]
                                self.address = ' '.join(self.address_arr)
                            else:
                                self.address = address
                                self.address_arr = address_arr
                            return name
                        elif child_node.level == 3 and not word:
                            name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                           province=self.province,
                                                                                           district=self.district)
                            self.address = address
                            self.address_arr = address_arr
                            return name
                    combined_word = (word + " " + child_node.word).strip() if word else child_node.word
                    if is_root:
                        if child_node.level == 2 and self.province != "" and self.province not in child_node.provinces:
                            return ""
                        elif child_node.level == 3 and (
                                self.district not in child_node.districts if self.district else False or self.province not in child_node.provinces):
                            if self.district != "" and self.province != "" and (
                                    self.district not in child_node.districts and self.province not in child_node.provinces):
                                return ""
                            elif self.district != "" and self.province == "" and (
                                    self.district not in child_node.districts):
                                return ""
                            elif self.district == "" and self.province != "" and (
                                    self.province not in child_node.provinces):
                                return ""
                        self.address_arr = self.address_arr[:len(self.address_arr) - self.word_count]
                        self.address = ' '.join(self.address_arr)
                        # self.address = self.address[:-len(combined_word) - 1]
                    return combined_word
                else:
                    if child_node.children:
                        word = self.search_trie(datas, child_node, remaining_s_arr, level, False)
                        combined_word = child_node.word if not word else f"{word} {child_node.word}"
                        if word == "":
                            if combined_word == "":
                                return ""
                            else:
                                name = combined_word
                                address_arr, address = None, None
                                if child_node.level == 2 and self.province not in child_node.provinces if self.province else False:
                                    name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                                   province=self.province)
                                elif child_node.level == 3 and (
                                        self.district not in child_node.districts if self.district else False or self.province not in child_node.provinces if self.province else False):
                                    name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                                   province=self.province,
                                                                                                   district=self.district)
                                self.address = address if address else self.address
                                self.address_arr = address_arr if address_arr else self.address_arr
                                self.word_count -= 1
                                return name
                        else:
                            key = str(word.split()[len(word.split()) - 1]).lower()
                            if key not in child_node.children:
                                return ""
                            if child_node.level == 2 and self.province not in child_node.children.get(key).provinces:
                                return ""
                            elif child_node.level == 3 and (self.district not in child_node.children.get(
                                    key).districts or self.province not in child_node.children.get(
                                key).provinces):
                                return ""
                        return combined_word.strip()
                    else:
                        if child_node.level == 2 and self.province not in child_node.provinces if self.province else False:
                            return ""
                        if child_node.level == 3 and (
                                self.district not in child_node.districts if self.district else False or self.province not in child_node.provinces if self.province else False):
                            return ""
                        self.address_arr = remaining_s_arr
                        self.word_count = 0
                        return child_node.word
            else:
                if is_root:
                    if s_lower in self.special:
                        return self.search_trie(datas, node, s_arr[:i], level, is_root)
                    name = self.search_kmp_trie(node, s_arr[i], is_root, True)
                    if not name:
                        name = self.search_kmp_trie(node, ' '.join(s_arr), is_root, False)
                        # return name
                        if not name or name == "":
                            name, address_arr, address = None, s_arr, self.address
                            if level == 1:
                                name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level)
                                self.address = address
                                self.address_arr = address_arr
                            elif level == 2:
                                name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                               province=self.province)
                                if not name:
                                    self.address_arr = self.address_arr[:len(self.address_arr) - self.word_count]
                                    self.address = ' '.join(self.address_arr)
                                else:
                                    self.address = address
                                    self.address_arr = address_arr
                            elif level == 3:
                                name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                               province=self.province,
                                                                                               district=self.district)
                                self.address = address
                                self.address_arr = address_arr
                            return name
                        if len(name) < len(s_arr[i]):
                            if level == 1:
                                name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level)
                                self.address = address
                                self.address_arr = address_arr
                            elif level == 2:
                                name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                               province=self.province)
                                if not name:
                                    self.address_arr = self.address_arr[:len(self.address_arr) - self.word_count]
                                    self.address = ' '.join(self.address_arr)
                                else:
                                    self.address = address
                                    self.address_arr = address_arr
                            elif level == 3:
                                name, address_arr, address = self.search_minimum_edit_distance(datas, s_arr, level,
                                                                                               province=self.province,
                                                                                               district=self.district)
                                self.address = address
                                self.address_arr = address_arr
                            return name
                    self.address_arr = s_arr[:i]
                    return name
                else:
                    return self.search_kmp_trie(node, s_arr[i], is_root, True)
            i -= 1
        return ""

    def preprocess(self, text):
        if ',,' in text or ', ,' in text:
            self.has_districts = False
        words = text.replace(",", " ").rstrip('.').split()
        new_word = self.ABBREVIATIONS.get(words[len(words) - 1])
        words[len(words) - 1] = new_word if new_word else words[len(words) - 1]
        return ' '.join([self.CORRECTED_VIETNAMESE_CHARS.get(word, word) for word in words])

    # -------------------------------------------------------------------------
    def process(self, s: str) -> dict:
        """
        Quy trình chính: chuẩn hóa địa chỉ, tách thành mảng từ,
        tìm kiếm province, district, ward theo thứ tự và trả về kết quả.
        """
        self.init_process()

        self.address = self.normalize_text(self.preprocess(s)).rstrip()
        self.address_arr = self.address.split()

        # Tìm kiếm theo thứ tự: province -> district -> ward
        self.word_count = 0
        self.province = self.search_trie(self.provinces, self.province_node, self.address_arr, 1, True)
        self.finish = False
        # Nếu quá trình cắt từ thay đổi độ dài mảng, cập nhật lại address_arr
        base_arr = self.address.split()
        self.address_arr = base_arr if len(base_arr) < len(self.address_arr) else self.address_arr

        if self.has_districts:
            self.word_count = 0
            self.district = self.search_trie(self.districts, self.district_node, self.address_arr, 2, True)
            self.finish = False
            base_arr = self.address.split()
            self.address_arr = base_arr if len(base_arr) < len(self.address_arr) else self.address_arr
        self.has_districts = True

        self.word_count = 0
        self.ward = self.search_trie(self.wards, self.ward_node, self.address_arr, 3, True)
        return {
            "province": self.province if self.province and self.province != "" else "",
            "district": self.district if self.district and self.district != "" else "",
            "ward": self.ward if self.ward and self.ward != "" else "",
        }

# NOTE: DO NOT change this cell
# This cell is for downloading private test
!rm -rf test.json
# this link is public test
!gdown --fuzzy https://drive.google.com/file/d/1PBt3U9I3EH885CDhcXspebyKI5Vw6uLB/view?usp=sharing -O test.json

# CORRECT TESTS
groups_province = {}
groups_district = {'hòa bình': ['Hoà Bình', 'Hòa Bình'], 'kbang': ['Kbang', 'KBang'], 'quy nhơn': ['Qui Nhơn', 'Quy Nhơn']}
groups_ward = {'ái nghĩa': ['ái Nghĩa', 'Ái Nghĩa'], 'ái quốc': ['ái Quốc', 'Ái Quốc'], 'ái thượng': ['ái Thượng', 'Ái Thượng'], 'ái tử': ['ái Tử', 'Ái Tử'], 'ấm hạ': ['ấm Hạ', 'Ấm Hạ'], 'an ấp': ['An ấp', 'An Ấp'], 'ẳng cang': ['ẳng Cang', 'Ẳng Cang'], 'ẳng nưa': ['ẳng Nưa', 'Ẳng Nưa'], 'ẳng tở': ['ẳng Tở', 'Ẳng Tở'], 'an hòa': ['An Hoà', 'An Hòa'], 'ayun': ['Ayun', 'AYun'], 'bắc ái': ['Bắc ái', 'Bắc Ái'], 'bảo ái': ['Bảo ái', 'Bảo Ái'], 'bình hòa': ['Bình Hoà', 'Bình Hòa'], 'châu ổ': ['Châu ổ', 'Châu Ổ'], 'chư á': ['Chư á', 'Chư Á'], 'chư rcăm': ['Chư Rcăm', 'Chư RCăm'], 'cộng hòa': ['Cộng Hoà', 'Cộng Hòa'], 'cò nòi': ['Cò  Nòi', 'Cò Nòi'], 'đại ân 2': ['Đại Ân  2', 'Đại Ân 2'], 'đak ơ': ['Đak ơ', 'Đak Ơ'], "đạ m'ri": ["Đạ M'ri", "Đạ M'Ri"], 'đông hòa': ['Đông Hoà', 'Đông Hòa'], 'đồng ích': ['Đồng ích', 'Đồng Ích'], 'hải châu i': ['Hải Châu  I', 'Hải Châu I'], 'hải hòa': ['Hải Hoà', 'Hải Hòa'], 'hành tín đông': ['Hành Tín  Đông', 'Hành Tín Đông'], 'hiệp hòa': ['Hiệp Hoà', 'Hiệp Hòa'], 'hòa bắc': ['Hoà Bắc', 'Hòa Bắc'], 'hòa bình': ['Hoà Bình', 'Hòa Bình'], 'hòa châu': ['Hoà Châu', 'Hòa Châu'], 'hòa hải': ['Hoà Hải', 'Hòa Hải'], 'hòa hiệp trung': ['Hoà Hiệp Trung', 'Hòa Hiệp Trung'], 'hòa liên': ['Hoà Liên', 'Hòa Liên'], 'hòa lộc': ['Hoà Lộc', 'Hòa Lộc'], 'hòa lợi': ['Hoà Lợi', 'Hòa Lợi'], 'hòa long': ['Hoà Long', 'Hòa Long'], 'hòa mạc': ['Hoà Mạc', 'Hòa Mạc'], 'hòa minh': ['Hoà Minh', 'Hòa Minh'], 'hòa mỹ': ['Hoà Mỹ', 'Hòa Mỹ'], 'hòa phát': ['Hoà Phát', 'Hòa Phát'], 'hòa phong': ['Hoà Phong', 'Hòa Phong'], 'hòa phú': ['Hoà Phú', 'Hòa Phú'], 'hòa phước': ['Hoà Phước', 'Hòa Phước'], 'hòa sơn': ['Hoà Sơn', 'Hòa Sơn'], 'hòa tân': ['Hoà Tân', 'Hòa Tân'], 'hòa thuận': ['Hoà Thuận', 'Hòa Thuận'], 'hòa tiến': ['Hoà Tiến', 'Hòa Tiến'], 'hòa trạch': ['Hoà Trạch', 'Hòa Trạch'], 'hòa vinh': ['Hoà Vinh', 'Hòa Vinh'], 'hương hòa': ['Hương Hoà', 'Hương Hòa'], 'ích hậu': ['ích Hậu', 'Ích Hậu'], 'ít ong': ['ít Ong', 'Ít Ong'], 'khánh hòa': ['Khánh Hoà', 'Khánh Hòa'], 'krông á': ['Krông Á', 'KRông á'], 'lộc hòa': ['Lộc Hoà', 'Lộc Hòa'], 'minh hòa': ['Minh Hoà', 'Minh Hòa'], 'mường ải': ['Mường ải', 'Mường Ải'], 'mường ẳng': ['Mường ẳng', 'Mường Ẳng'], 'nậm ét': ['Nậm ét', 'Nậm Ét'], 'nam hòa': ['Nam Hoà', 'Nam Hòa'], 'na ư': ['Na ư', 'Na Ư'], 'ngã sáu': ['Ngã sáu', 'Ngã Sáu'], 'nghi hòa': ['Nghi Hoà', 'Nghi Hòa'], 'nguyễn úy': ['Nguyễn Uý', 'Nguyễn úy', 'Nguyễn Úy'], 'nhân hòa': ['Nhân Hoà', 'Nhân Hòa'], 'nhơn hòa': ['Nhơn Hoà', 'Nhơn Hòa'], 'nhơn nghĩa a': ['Nhơn nghĩa A', 'Nhơn Nghĩa A'], 'phúc ứng': ['Phúc ứng', 'Phúc Ứng'], 'phước hòa': ['Phước Hoà', 'Phước Hòa'], 'sơn hóa': ['Sơn Hoá', 'Sơn Hóa'], 'tạ an khương đông': ['Tạ An Khương  Đông', 'Tạ An Khương Đông'], 'tạ an khương nam': ['Tạ An Khương  Nam', 'Tạ An Khương Nam'], 'tăng hòa': ['Tăng Hoà', 'Tăng Hòa'], 'tân hòa': ['Tân Hoà', 'Tân Hòa'], 'tân hòa thành': ['Tân Hòa  Thành', 'Tân Hòa Thành'], 'tân khánh trung': ['Tân  Khánh Trung', 'Tân Khánh Trung'], 'tân lợi': ['Tân lợi', 'Tân Lợi'], 'thái hòa': ['Thái Hoà', 'Thái Hòa'], 'thiết ống': ['Thiết ống', 'Thiết Ống'], 'thuận hòa': ['Thuận Hoà', 'Thuận Hòa'], 'thượng ấm': ['Thượng ấm', 'Thượng Ấm'], 'thụy hương': ['Thuỵ Hương', 'Thụy Hương'], 'thủy xuân': ['Thuỷ Xuân', 'Thủy Xuân'], 'tịnh ấn đông': ['Tịnh ấn Đông', 'Tịnh Ấn Đông'], 'tịnh ấn tây': ['Tịnh ấn Tây', 'Tịnh Ấn Tây'], 'triệu ái': ['Triệu ái', 'Triệu Ái'], 'triệu ẩu': ['Triệu ẩu', 'Triệu Ẩu'], 'trung hòa': ['Trung Hoà', 'Trung Hòa'], 'trung ý': ['Trung ý', 'Trung Ý'], 'tùng ảnh': ['Tùng ảnh', 'Tùng Ảnh'], 'úc kỳ': ['úc Kỳ', 'Úc Kỳ'], 'ứng hòe': ['ứng Hoè', 'Ứng Hoè'], 'vĩnh hòa': ['Vĩnh Hoà', 'Vĩnh Hòa'], 'vũ hòa': ['Vũ Hoà', 'Vũ Hòa'], 'xuân ái': ['Xuân ái', 'Xuân Ái'], 'xuân áng': ['Xuân áng', 'Xuân Áng'], 'xuân hòa': ['Xuân Hoà', 'Xuân Hòa'], 'xuất hóa': ['Xuất Hoá', 'Xuất Hóa'], 'ỷ la': ['ỷ La', 'Ỷ La']}
groups_ward.update({1: ['1', '01'], 2: ['2', '02'], 3: ['3', '03'], 4: ['4', '04'], 5: ['5', '05'], 6: ['6', '06'], 7: ['7', '07'], 8: ['8', '08'], 9: ['9', '09']})
def to_same(groups):
    same = {ele: k for k, v in groups.items() for ele in v}
    return same
same_province = to_same(groups_province)
same_district = to_same(groups_district)
same_ward = to_same(groups_ward)
def normalize(text, same_dict):
    return same_dict.get(text, text)

TEAM_NAME = 'DEFAULT_NAME'  # This should be your team name
EXCEL_FILE = f'{TEAM_NAME}.xlsx'

import json
import time
with open('test.json') as f:
    data = json.load(f)

summary_only = True
df = []
solution = Solution()
timer = []
correct = 0
for test_idx, data_point in enumerate(data):
    address = data_point["text"]

    ok = 0
    try:
        answer = data_point["result"]
        answer["province_normalized"] = normalize(answer["province"], same_province)
        answer["district_normalized"] = normalize(answer["district"], same_district)
        answer["ward_normalized"] = normalize(answer["ward"], same_ward)

        start = time.perf_counter_ns()
        result = solution.process(address)
        finish = time.perf_counter_ns()
        timer.append(finish - start)
        result["province_normalized"] = normalize(result["province"], same_province)
        result["district_normalized"] = normalize(result["district"], same_district)
        result["ward_normalized"] = normalize(result["ward"], same_ward)

        province_correct = int(answer["province_normalized"] == result["province_normalized"])
        district_correct = int(answer["district_normalized"] == result["district_normalized"])
        ward_correct = int(answer["ward_normalized"] == result["ward_normalized"])
        ok = province_correct + district_correct + ward_correct

        df.append([
            test_idx,
            address,
            answer["province"],
            result["province"],
            answer["province_normalized"],
            result["province_normalized"],
            province_correct,
            answer["district"],
            result["district"],
            answer["district_normalized"],
            result["district_normalized"],
            district_correct,
            answer["ward"],
            result["ward"],
            answer["ward_normalized"],
            result["ward_normalized"],
            ward_correct,
            ok,
            timer[-1] / 1_000_000_000,
        ])
    except Exception as e:
        print(e)
        print(f"{answer = }")
        print(f"{result = }")
        df.append([
            test_idx,
            address,
            answer["province"],
            "EXCEPTION",
            answer["province_normalized"],
            "EXCEPTION",
            0,
            answer["district"],
            "EXCEPTION",
            answer["district_normalized"],
            "EXCEPTION",
            0,
            answer["ward"],
            "EXCEPTION",
            answer["ward_normalized"],
            "EXCEPTION",
            0,
            0,
            0,
        ])
        # any failure count as a zero correct
        pass
    correct += ok


    if not summary_only:
        # responsive stuff
        print(f"Test {test_idx:5d}/{len(data):5d}")
        print(f"Correct: {ok}/3")
        print(f"Time Executed: {timer[-1] / 1_000_000_000:.4f}")


print(f"-"*30)
total = len(data) * 3
score_scale_10 = round(correct / total * 10, 2)
if len(timer) == 0:
    timer = [0]
max_time_sec = round(max(timer) / 1_000_000_000, 4)
avg_time_sec = round((sum(timer) / len(timer)) / 1_000_000_000, 4)

import pandas as pd

df2 = pd.DataFrame(
    [[correct, total, score_scale_10, max_time_sec, avg_time_sec]],
    columns=['correct', 'total', 'score / 10', 'max_time_sec', 'avg_time_sec',],
)

columns = [
    'ID',
    'text',
    'province',
    'province_student',
    'province_normalized',
    'province_student_normalized',
    'province_correct',
    'district',
    'district_student',
    'district_normalized',
    'district_student_normalized',
    'district_correct',
    'ward',
    'ward_student',
    'ward_normalized',
    'ward_student_normalized',
    'ward_correct',
    'total_correct',
    'time_sec',
]

df = pd.DataFrame(df)
df.columns = columns

print(f'{TEAM_NAME = }')
print(f'{EXCEL_FILE = }')
print(df2)

!pip install xlsxwriter
writer = pd.ExcelWriter(EXCEL_FILE, engine='xlsxwriter')
df2.to_excel(writer, index=False, sheet_name='summary')
df.to_excel(writer, index=False, sheet_name='details')
writer.close()