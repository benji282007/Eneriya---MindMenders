import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import json, os, re, time, cv2, platform, threading, math

# Optional dependencies for Memory Assistant (face recognition only)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
except ImportError:
    HAS_FACE_RECOGNITION = False

ctk.set_appearance_mode("dark")

DB_FILE="database.json"
IMAGE_FOLDER="images"
ENCODINGS_CACHE_FILE = "face_encodings_cache.json"

# ---------------- DATABASE ----------------
def load_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE,"w") as f:
            json.dump({"people":[]},f)
    try:
        with open(DB_FILE,"r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {"people":[]}

def save_db(data):
    with open(DB_FILE,"w") as f:
        json.dump(data,f,indent=4)

# ---------------- MEMORY ASSISTANT HELPERS ----------------
def _load_encodings_cache():
    """Load cached face encodings from disk. Returns dict: path -> {"mtime": float, "encoding": list}."""
    if not os.path.exists(ENCODINGS_CACHE_FILE):
        return {}
    try:
        with open(ENCODINGS_CACHE_FILE, "r") as f:
            data = json.load(f)
        return {k: {"mtime": v["mtime"], "encoding": np.array(v["encoding"]) if HAS_NUMPY else v["encoding"]} for k, v in data.items()}
    except Exception:
        return {}

def _save_encodings_cache(cache):
    """Save encodings cache to disk. cache: path -> {"mtime": float, "encoding": array or list}."""
    try:
        data = {}
        for path, v in cache.items():
            enc = v["encoding"]
            data[path] = {"mtime": v["mtime"], "encoding": enc.tolist() if hasattr(enc, "tolist") else list(enc)}
        with open(ENCODINGS_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=0)
    except Exception:
        pass

def load_known_faces_from_app_db(db):
    """Returns (encodings_list, names_list, relations_dict, metadata_dict). Uses a disk cache so 100+ users don't recompute encodings every run."""
    encodings_list, names_list = [], []
    relations_dict, metadata_dict = {}, {}
    people = db.get("people", [])
    if not HAS_FACE_RECOGNITION:
        return encodings_list, names_list, relations_dict, metadata_dict
    cache = _load_encodings_cache()
    cache_updated = False
    for person in people:
        name = person.get("name", "").strip()
        rel = person.get("relation", "").strip() or "Stranger"
        img_path = person.get("image") or ""
        if not name or not img_path or not os.path.exists(img_path):
            continue
        try:
            mtime = os.path.getmtime(img_path)
            use_cache = img_path in cache and cache[img_path]["mtime"] == mtime
            if use_cache:
                enc = cache[img_path]["encoding"]
                if HAS_NUMPY and isinstance(enc, np.ndarray) and enc.size == 128:
                    encodings_list.append(enc)
                    names_list.append(name)
                    relations_dict[name] = rel
                    metadata_dict[name] = img_path
                else:
                    use_cache = False
            if not use_cache:
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    enc = encodings[0]
                    encodings_list.append(enc)
                    names_list.append(name)
                    relations_dict[name] = rel
                    metadata_dict[name] = img_path
                    cache[img_path] = {"mtime": mtime, "encoding": enc}
                    cache_updated = True
        except Exception:
            continue
    if cache_updated:
        valid_paths = {p.get("image") or "" for p in people}
        pruned = {p: cache[p] for p in cache if p in valid_paths}
        _save_encodings_cache(pruned)
    return encodings_list, names_list, relations_dict, metadata_dict

def is_valid_email(email):
    if not email or len(email) > 254:
        return False
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email.strip()) is not None

def is_valid_password(password):
    if not password or len(password) < 8:
        return False
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_letter and has_digit

# ---------------- BUTTON STYLE ----------------
def btn(parent,text,cmd,w=200):
    return ctk.CTkButton(
        parent,text=text,command=cmd,
        width=w,height=48,
        corner_radius=28,
        fg_color="white",
        text_color="black",
        hover_color="#dddddd"
    )

# ---------------- APP ----------------
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.geometry("1200x720")
        self.configure(fg_color="black")
        self.title("MindMenders")

        self.db=load_db()
        self.user_name=""

        self.frames={}
        for F in (Splash,Login,Home,AddPerson,Detected,ProfileView,EditProfile,MemoryAssistant):
            frame=F(self)
            self.frames[F]=frame
            frame.place(relwidth=1,relheight=1)

        self.show(Splash)

    def show(self,page):
        self.frames[page].tkraise()
        if hasattr(self.frames[page], "on_show"):
            self.frames[page].on_show()

# ---------------- SPLASH ----------------
class Splash(ctk.CTkFrame):
    def __init__(self,app):
        super().__init__(app,fg_color="black")

        ctk.CTkLabel(self,text="MindMenders",
            font=ctk.CTkFont(size=60,weight="bold")
        ).place(relx=0.5,rely=0.45,anchor="center")

        btn(self,"Login",
            lambda: self.master.show(Login),260).place(relx=0.5,rely=0.6,anchor="center")

# ---------------- LOGIN ----------------
class Login(ctk.CTkFrame):
    def __init__(self,app):
        super().__init__(app,fg_color="black")
        self.app=app

        box=ctk.CTkFrame(self,fg_color="black")
        box.place(relx=0.5,rely=0.5,anchor="center")

        ctk.CTkLabel(box,text="Sign In",
            font=ctk.CTkFont(size=42,weight="bold")).pack(pady=20)

        self.name=ctk.CTkEntry(box,width=420,placeholder_text="Name")
        self.name.pack(pady=10)

        self.email=ctk.CTkEntry(box,width=420,placeholder_text="Email")
        self.email.pack(pady=10)

        self.password=ctk.CTkEntry(box,width=420,show="*",placeholder_text="Password")
        self.password.pack(pady=10)

        ctk.CTkLabel(box,text="Email must be valid; password: 8+ characters with letters and numbers.",
                     font=ctk.CTkFont(size=12),text_color="gray").pack(pady=(0,10))

        btn(box,"Sign In",self.login,320).pack(pady=20)

        ctk.CTkButton(box, text="← Back to welcome", command=lambda: self.app.show(Splash),
                      fg_color="transparent", text_color="#888", hover_color="#222", width=180, height=36).pack(pady=10)

    def login(self):
        email = self.email.get().strip()
        password = self.password.get()
        if not is_valid_email(email):
            messagebox.showerror("Invalid Email", "Please enter a valid email address (e.g. name@example.com).")
            return
        if not is_valid_password(password):
            messagebox.showerror(
                "Invalid Password",
                "Password must be at least 8 characters and contain both letters and numbers."
            )
            return
        self.app.user_name = self.name.get().strip()
        self.app.frames[Home].update_name()
        self.app.show(Home)

# ---------------- HOME ----------------
class Home(ctk.CTkFrame):
    def __init__(self,app):
        super().__init__(app,fg_color="black")
        self.app=app

        box=ctk.CTkFrame(self,fg_color="black")
        box.place(relx=0.5,rely=0.5,anchor="center")

        self.label=ctk.CTkLabel(box,
            font=ctk.CTkFont(size=46,weight="bold"))
        self.label.pack(pady=40)

        btn(box,"Add Person",
            lambda: self.app.show(AddPerson),320).pack(pady=15)

        btn(box,"Detected Persons",
            lambda: self.app.show(Detected),320).pack(pady=15)

        btn(box,"Memory Assistant",lambda: self.app.show(MemoryAssistant),320).pack(pady=15)

        btn(box,"Sign out",self.sign_out,200).pack(pady=20)

    def sign_out(self):
        self.app.user_name = ""
        self.app.show(Login)

    def update_name(self):
        self.label.configure(text=f"Welcome, {self.app.user_name}")

# ---------------- ADD PERSON ----------------
class AddPerson(ctk.CTkFrame):
    def __init__(self,app):
        super().__init__(app,fg_color="black")
        self.app=app
        self.cap=None
        self.image_path=None

        btn(self,"← Back",self._go_back,140)\
            .pack(anchor="nw",padx=20,pady=20)

        self.scroll = ctk.CTkScrollableFrame(self, fg_color="black")
        self.scroll.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.video=ctk.CTkLabel(self.scroll,text="Photo Preview",
                                width=520,height=340,fg_color="#111")
        self.video.pack(pady=15)

        row=ctk.CTkFrame(self.scroll,fg_color="black")
        row.pack()

        btn(row,"Open Camera",self.start_camera,150).pack(side="left",padx=6)
        btn(row,"Capture",self.capture,150).pack(side="left",padx=6)
        btn(row,"Upload",self.upload,150).pack(side="left",padx=6)

        name_row = ctk.CTkFrame(self.scroll, fg_color="black")
        name_row.pack(fill="x", pady=(14, 0))
        ctk.CTkLabel(name_row, text="Name", font=ctk.CTkFont(size=13), text_color="#888").pack(anchor="w", pady=(0, 3))
        self.name = ctk.CTkEntry(name_row, width=420)
        self.name.pack(fill="x", pady=(0, 12))

        relation_row = ctk.CTkFrame(self.scroll, fg_color="black")
        relation_row.pack(fill="x")
        ctk.CTkLabel(relation_row, text="Relation", font=ctk.CTkFont(size=13), text_color="#888").pack(anchor="w", pady=(0, 3))
        self.relation = ctk.CTkEntry(relation_row, width=420)
        self.relation.pack(fill="x", pady=(0, 12))

        notes_row = ctk.CTkFrame(self.scroll, fg_color="black")
        notes_row.pack(fill="x")
        ctk.CTkLabel(notes_row, text="Notes", font=ctk.CTkFont(size=13), text_color="#888").pack(anchor="w", pady=(0, 3))
        self.notes = ctk.CTkTextbox(notes_row, width=420, height=90)
        self.notes.pack(fill="x", pady=(0, 10))

        btn(self.scroll,"Save",self.save,260).pack(pady=15)

    def on_show(self):
        """Clear form and preview when opening Add Person again."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.image_path = None
        self.name.delete(0, "end")
        self.relation.delete(0, "end")
        self.notes.delete("1.0", "end")
        self.video.configure(image=None, text="Photo Preview")
        self.video.image = None
        if hasattr(self.video, "_pil_image"):
            self.video._pil_image = None
        self.focus_set()
        self.after(10, self._ensure_placeholders_visible)

    def _ensure_placeholders_visible(self):
        """Keep focus off entries so placeholders stay visible after screen is drawn."""
        if self.winfo_exists():
            self.focus_set()

    def _go_back(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.app.show(Home)

    def start_camera(self):
        if self.cap and self.cap.isOpened():
            return
        # Try platform-appropriate backend, then fallback to default
        backends = []
        if platform.system() == "Darwin":
            backends = [(0, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)), (0, cv2.CAP_ANY)]
        elif platform.system() == "Windows":
            backends = [(0, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)), (0, cv2.CAP_ANY)]
        else:
            backends = [(0, cv2.CAP_ANY), (0, getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))]
        self.cap = None
        for idx, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror(
                "Camera",
                "Could not open camera. Check that:\n"
                "• No other app is using the camera\n"
                "• Camera access is allowed for this app (System Settings → Privacy)\n"
                "• The camera is connected."
            )
            return
        # Warm up: discard first few frames (often dark or invalid)
        for _ in range(5):
            self.cap.read()
        self.update_frame()

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret,frame=self.cap.read()
            if ret:
                frame=cv2.flip(frame,1)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                pil_img=Image.fromarray(rgb).resize((520,340))
                photo=ctk.CTkImage(light_image=pil_img,size=(520,340))
                self.video.configure(image=photo,text="")
                self.video.image=photo
                self.video._pil_image=pil_img
            self.after(20,self.update_frame)

    def capture(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Camera", "Open the camera first.")
            return
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Camera", "Could not capture frame.")
            return
        frame = cv2.flip(frame, 1)
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)
        path = f"{IMAGE_FOLDER}/cam_{int(time.time())}.jpg"
        cv2.imwrite(path, frame)
        self.image_path = path
        self.cap.release()
        self.cap = None
        # Show captured image in preview
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((520, 340))
        photo = ctk.CTkImage(light_image=pil_img, size=(520, 340))
        self.video.configure(image=photo, text="")
        self.video.image = photo
        self.video._pil_image = pil_img

    def upload(self):
        file=filedialog.askopenfilename(
            filetypes=[("Images","*.png *.jpg *.jpeg")])
        if file:
            self.image_path=file
            pil_img=Image.open(file).convert("RGB").resize((520,340))
            photo=ctk.CTkImage(light_image=pil_img,size=(520,340))
            self.video.configure(image=photo,text="")
            self.video.image=photo
            self.video._pil_image=pil_img

    def save(self):
        name = self.name.get().strip()
        if not name:
            messagebox.showwarning("Save", "Please enter a name.")
            return
        if not self.image_path or not os.path.exists(self.image_path):
            messagebox.showwarning("Save", "Please add a photo (camera or upload).")
            return
        self.app.db["people"].append({
            "name": name,
            "relation": self.relation.get().strip(),
            "notes": self.notes.get("1.0", "end").strip(),
            "image": self.image_path
        })
        save_db(self.app.db)
        self.app.show(Detected)
        self.app.frames[Detected].refresh()

# ---------------- DETECTED ----------------
class Detected(ctk.CTkFrame):
    def __init__(self,app):
        super().__init__(app,fg_color="black")
        self.app=app

        btn(self,"← Back",lambda: self.app.show(Home),140)\
            .pack(anchor="nw",padx=20,pady=20)

        self.scroll=ctk.CTkScrollableFrame(self)
        self.scroll.pack(expand=True,fill="both",padx=40,pady=20)

    def on_show(self):
        self.refresh()

    def refresh(self):
        for w in self.scroll.winfo_children():
            w.destroy()
        people = self.app.db.get("people", [])
        def make_open_cmd(p):
            def _open():
                self.open_profile(p)
            return _open

        for idx, person in enumerate(people):
            open_cmd = make_open_cmd(person)
            card = ctk.CTkFrame(self.scroll, fg_color="#111", corner_radius=20, cursor="hand2")
            card.pack(fill="x", pady=10)
            card.bind("<Button-1>", lambda e, cmd=open_cmd: cmd())

            # Thumbnail (always show: image or placeholder with initial/name)
            thumb_frame = ctk.CTkFrame(card, width=65, height=65, fg_color="#222", corner_radius=12)
            thumb_frame.pack(side="left", padx=15, pady=10)
            thumb_frame.pack_propagate(False)
            img_path = person.get("image") or ""
            if img_path and os.path.exists(img_path):
                try:
                    pil_img = Image.open(img_path).convert("RGB").resize((65, 65))
                    photo = ctk.CTkImage(light_image=pil_img, size=(65, 65))
                    img_lbl = ctk.CTkLabel(thumb_frame, image=photo, text="")
                    img_lbl.image = photo
                    img_lbl._pil_image = pil_img  # keep reference so image is not GC'd
                    img_lbl.place(relx=0.5, rely=0.5, anchor="center")
                except Exception:
                    initial = (person.get("name", "?") or "?")[0].upper()
                    ctk.CTkLabel(thumb_frame, text=initial, font=ctk.CTkFont(size=24, weight="bold")).place(relx=0.5, rely=0.5, anchor="center")
            else:
                initial = (person.get("name", "?") or "?")[0].upper()
                ctk.CTkLabel(thumb_frame, text=initial, font=ctk.CTkFont(size=24, weight="bold")).place(relx=0.5, rely=0.5, anchor="center")

            name_str = person.get("name", "")
            relation_str = person.get("relation", "")
            text = ctk.CTkLabel(
                card,
                text=f"{name_str} — {relation_str}",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            text.pack(side="left", padx=10, pady=10)

            open_btn = btn(card, "Open", open_cmd, 120)
            open_btn.pack(side="right", padx=15, pady=10)
        self.scroll.update()

    def open_profile(self,person):
        self.app.frames[ProfileView].load(person)
        self.app.show(ProfileView)

# ---------------- PROFILE VIEW ----------------
class ProfileView(ctk.CTkFrame):
    def __init__(self,app):
        super().__init__(app,fg_color="black")
        self.app=app

        btn(self,"← Back",lambda: self.app.show(Detected),140)\
            .pack(anchor="nw",padx=20,pady=20)

        self.img=ctk.CTkLabel(self,text="")
        self.img.pack(pady=20)

        ctk.CTkLabel(self, text="Name", font=ctk.CTkFont(size=14), text_color="#888").pack(pady=(0, 4))
        self.name=ctk.CTkLabel(self,font=ctk.CTkFont(size=38,weight="bold"))
        self.name.pack(pady=(0, 12))

        ctk.CTkLabel(self, text="Relationship", font=ctk.CTkFont(size=14), text_color="#888").pack(pady=(0, 4))
        self.rel=ctk.CTkLabel(self,font=ctk.CTkFont(size=20))
        self.rel.pack(pady=(0, 16))

        ctk.CTkLabel(self, text="Bio", font=ctk.CTkFont(size=14), text_color="#888").pack(pady=(0, 4))
        self.notes=ctk.CTkLabel(self,wraplength=620,font=ctk.CTkFont(size=18))
        self.notes.pack(pady=(0, 20))

        row=ctk.CTkFrame(self,fg_color="black")
        row.pack(pady=15)
        btn(row,"Edit Profile",self.edit,200).pack(side="left",padx=8)
        btn(row,"Delete",self.delete_person,200).pack(side="left",padx=8)

    def load(self,person):
        self.person=person
        img_path = person.get("image") or ""
        if img_path and os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert("RGB").resize((380, 380))
                photo = ctk.CTkImage(light_image=pil_img, size=(380, 380))
                self.img.configure(image=photo, text="")
                self.img.image = photo
                self.img._pil_image = pil_img  # keep reference
            except Exception:
                self.img.configure(image=None, text="No photo")
                self.img.image = None
                self.img._pil_image = None
        else:
            self.img.configure(image=None, text="No photo")
            self.img.image = None
            self.img._pil_image = None
        self.name.configure(text=person.get("name", ""))
        self.rel.configure(text=person.get("relation", ""))
        self.notes.configure(text=person.get("notes", ""))

    def edit(self):
        self.app.frames[EditProfile].load(self.person)
        self.app.show(EditProfile)

    def delete_person(self):
        if not messagebox.askyesno("Delete", f"Delete {self.person.get('name', 'this person')}? This cannot be undone."):
            return
        try:
            self.app.db["people"].remove(self.person)
        except ValueError:
            pass
        save_db(self.app.db)
        self.app.frames[Detected].refresh()
        self.app.show(Detected)

# ---------------- CAPTURE PHOTO DIALOG (for Edit Profile) ----------------
class _CapturePhotoDialog(ctk.CTkToplevel):
    def __init__(self, app, edit_profile_frame):
        super().__init__(app)
        self.edit_profile_frame = edit_profile_frame
        self.cap = None
        self.title("Capture photo")
        self.geometry("540x420")
        self.configure(fg_color="black")

        self.video = ctk.CTkLabel(self, text="Open camera to capture", width=500, height=300, fg_color="#111")
        self.video.pack(pady=15)

        row = ctk.CTkFrame(self, fg_color="black")
        row.pack(pady=10)
        btn(row, "Open Camera", self._start_camera, 140).pack(side="left", padx=6)
        btn(row, "Capture", self._capture, 140).pack(side="left", padx=6)
        btn(row, "Cancel", self._close, 100).pack(side="left", padx=6)

        self.protocol("WM_DELETE_WINDOW", self._close)

    def _start_camera(self):
        if self.cap and self.cap.isOpened():
            return
        backends = []
        if platform.system() == "Darwin":
            backends = [(0, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)), (0, cv2.CAP_ANY)]
        elif platform.system() == "Windows":
            backends = [(0, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)), (0, cv2.CAP_ANY)]
        else:
            backends = [(0, cv2.CAP_ANY), (0, getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))]
        self.cap = None
        for idx, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Camera", "Could not open camera.")
            return
        for _ in range(5):
            self.cap.read()
        self._update_frame()

    def _update_frame(self):
        if self.cap and self.cap.isOpened() and self.winfo_exists():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb).resize((500, 300))
                photo = ctk.CTkImage(light_image=pil_img, size=(500, 300))
                self.video.configure(image=photo, text="")
                self.video.image = photo
                self.video._pil_image = pil_img
            self.after(20, self._update_frame)

    def _capture(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showwarning("Camera", "Open the camera first.")
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        if not os.path.exists(IMAGE_FOLDER):
            os.makedirs(IMAGE_FOLDER)
        path = f"{IMAGE_FOLDER}/edit_cam_{int(time.time())}.jpg"
        cv2.imwrite(path, frame)
        self.edit_profile_frame.person["image"] = path
        self.edit_profile_frame._show_photo()
        self._close()

    def _close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.destroy()

# ---------------- EDIT PROFILE ----------------
class EditProfile(ctk.CTkFrame):
    def __init__(self,app):
        super().__init__(app,fg_color="black")
        self.app=app

        btn(self,"← Back",lambda: self.app.show(ProfileView),140)\
            .pack(anchor="nw",padx=20,pady=20)

        self.photo_label=ctk.CTkLabel(self,text="Photo",width=200,height=200,fg_color="#111",corner_radius=12)
        self.photo_label.pack(pady=10)
        photo_row=ctk.CTkFrame(self,fg_color="black")
        photo_row.pack(pady=6)
        btn(photo_row,"Upload photo",self.change_photo,160).pack(side="left",padx=6)
        btn(photo_row,"Capture photo",self.capture_photo,160).pack(side="left",padx=6)

        self.name=ctk.CTkEntry(self,width=420,height=44,font=ctk.CTkFont(size=16))
        self.name.pack(pady=8)

        self.rel=ctk.CTkEntry(self,width=420,height=44,font=ctk.CTkFont(size=16))
        self.rel.pack(pady=8)

        self.notes=ctk.CTkTextbox(self,width=420,height=120,font=ctk.CTkFont(size=14))
        self.notes.pack(pady=10)

        btn(self,"Save Changes",self.save,240).pack(pady=12)

    def load(self,person):
        self.person=person
        self._show_photo()
        self.name.delete(0,"end")
        self.name.insert(0,person.get("name",""))
        self.rel.delete(0,"end")
        self.rel.insert(0,person.get("relation",""))
        self.notes.delete("1.0","end")
        self.notes.insert("1.0",person.get("notes",""))

    def _show_photo(self):
        img_path = self.person.get("image") or ""
        if img_path and os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert("RGB").resize((200, 200))
                photo = ctk.CTkImage(light_image=pil_img, size=(200, 200))
                self.photo_label.configure(image=photo, text="")
                self.photo_label.image = photo
                self.photo_label._pil_image = pil_img
            except Exception:
                self.photo_label.configure(image=None, text="No photo")
                self.photo_label.image = None
        else:
            self.photo_label.configure(image=None, text="No photo")
            self.photo_label.image = None

    def change_photo(self):
        file = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not file:
            return
        self.person["image"] = file
        self._show_photo()

    def capture_photo(self):
        """Open a small window to capture photo from camera."""
        _CapturePhotoDialog(self.app, self)

    def save(self):
        self.person["name"]=self.name.get().strip()
        self.person["relation"]=self.rel.get().strip()
        self.person["notes"]=self.notes.get("1.0","end").strip()
        save_db(self.app.db)
        messagebox.showinfo("Saved","Profile Updated")
        self.app.frames[ProfileView].load(self.person)
        self.app.show(ProfileView)


# ---------------- MEMORY ASSISTANT (Live face recognition + voice) ----------------
RECOGNITION_TOLERANCE = 0.4
TRIGGER_THRESHOLD = 8
SAVE_DURATION = 5
PENDING_REGISTRATION_TIMEOUT_SEC = 300  # 5 minutes; auto-cancel if nothing done
MA_VIDEO_SIZE = (800, 500)

class MemoryAssistant(ctk.CTkFrame):
    def __init__(self, app):
        super().__init__(app, fg_color="black")
        self.app = app
        self.cap = None
        self.running = False
        self.current_frame = None
        self.detected_list = []
        self.known_encodings = []
        self.known_names = []
        self.known_relations = {}
        self.known_metadata = {}
        self.active_unknowns = {}
        self.next_id = 0
        self.pending_unknowns = []  # captured unknowns kept until registered or 5 min timeout
        self._pending_id_counter = 0
        self._last_sidebar_state = None
        self._pending_ids_shown = ()  # when non-empty, sidebar is frozen so user can type
        self._lock = threading.Lock()

        btn(self, "← Back", self._go_back, 140).pack(anchor="nw", padx=20, pady=20)

        content = ctk.CTkFrame(self, fg_color="black")
        content.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self.video_label = ctk.CTkLabel(content, text="Starting camera…", width=MA_VIDEO_SIZE[0], height=MA_VIDEO_SIZE[1], fg_color="#111", corner_radius=8)
        self.video_label.pack(side="left", padx=(0, 20), pady=10)

        sidebar = ctk.CTkFrame(content, width=320, fg_color="#0d0d0d", corner_radius=12)
        sidebar.pack(side="right", fill="y", padx=0, pady=10)
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="LIVE RECOGNITION", font=ctk.CTkFont(size=16, weight="bold"), text_color="#ffa500").pack(pady=16, padx=16, anchor="w")
        ctk.CTkLabel(sidebar, text="Detected people", font=ctk.CTkFont(size=12), text_color="#888").pack(pady=(0, 12), padx=16, anchor="w")

        self.sidebar_scroll = ctk.CTkScrollableFrame(sidebar, fg_color="transparent")
        self.sidebar_scroll.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def on_show(self):
        if not HAS_NUMPY or not HAS_FACE_RECOGNITION:
            messagebox.showinfo(
                "Memory Assistant",
                "Install required packages:\npip install numpy face_recognition"
            )
            return
        self._reload_known_faces()
        self.running = True
        self._pending_ids_shown = ()
        self._start_camera_then_run()
        self._poll_ui()

    def on_hide(self):
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def _go_back(self):
        self.on_hide()
        self.app.show(Home)

    def _reload_known_faces(self):
        enc, names, rels, meta = load_known_faces_from_app_db(self.app.db)
        with self._lock:
            self.known_encodings = enc
            self.known_names = names
            self.known_relations = rels
            self.known_metadata = meta

    def _add_pending_unknown(self, crop):
        """Add captured unknown to sidebar (kept until registered via keyboard). Only one at a time."""
        if self.pending_unknowns:
            return
        pending_id = self._pending_id_counter
        self._pending_id_counter += 1
        self.pending_unknowns.append({
            "id": pending_id,
            "image": crop,
            "created_at": time.time(),
        })

    def _start_camera_then_run(self):
        """Open camera on main thread, then run one recognition frame (avoids cross-thread buffer crashes)."""
        if self.cap is not None and self.cap.isOpened():
            self.after(30, self._run_one_frame)
            return
        backends = []
        if platform.system() == "Darwin":
            backends = [(0, getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)), (0, cv2.CAP_ANY)]
        elif platform.system() == "Windows":
            backends = [(0, getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)), (0, cv2.CAP_ANY)]
        else:
            backends = [(0, cv2.CAP_ANY), (0, getattr(cv2, "CAP_V4L2", cv2.CAP_ANY))]
        self.cap = None
        for idx, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Memory Assistant", "Could not open camera.")
            return
        for _ in range(5):
            self.cap.read()
        self.after(0, self._run_one_frame)

    def _run_one_frame(self):
        """Run one recognition iteration on the main thread (no cross-thread numpy/OpenCV). Safe on Mac."""
        if not self.winfo_exists() or not self.running or not self.cap or not self.cap.isOpened():
            return
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.after(30, self._run_one_frame)
                return
            frame = cv2.flip(frame, 1).copy()
            h, w = int(MA_VIDEO_SIZE[1]), int(MA_VIDEO_SIZE[0])
            frame_resized = cv2.resize(frame, (w, h))
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            try:
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
            except Exception:
                with self._lock:
                    self.current_frame = frame_resized.copy()
                    self.detected_list = []
                self.after(30, self._run_one_frame)
                return

            color_safe, color_warn = (0, 255, 127), (71, 71, 255)
            current_frame_unidentified = []
            detected_list = []

            with self._lock:
                known_enc = list(self.known_encodings)
                known_names = list(self.known_names)
                known_relations = dict(self.known_relations)
                known_metadata = dict(self.known_metadata)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name, relation, ref_image = "Unknown", "Stranger", None
                t, r, b, l = top * 4, right * 4, bottom * 4, left * 4
                scale_x, scale_y = w / (frame.shape[1]), h / (frame.shape[0])
                td, rd, bd, ld = int(t * scale_y), int(r * scale_x), int(b * scale_y), int(l * scale_x)

                if known_enc:
                    face_distances = face_recognition.face_distance(known_enc, face_encoding)
                    best_idx = np.argmin(face_distances)
                    if face_distances[best_idx] < RECOGNITION_TOLERANCE:
                        name = known_names[best_idx]
                        relation = known_relations.get(name, "Known")
                        if name in known_metadata:
                            try:
                                ref_image = cv2.imread(known_metadata[name])
                                if ref_image is not None:
                                    ref_image = ref_image.copy()
                            except Exception:
                                pass

                if ref_image is None or ref_image.size == 0:
                    ref_image = frame[max(0, t):min(frame.shape[0], b), max(0, l):min(frame.shape[1], r)].copy()

                notes = ""
                if name != "Unknown":
                    for p in self.app.db.get("people", []):
                        if (p.get("name") or "").strip() == name:
                            notes = (p.get("notes") or "").strip()
                            break
                detected_list.append({"name": name, "rel": relation, "image": ref_image.copy(), "notes": notes})

                color = color_safe if relation != "Stranger" else color_warn
                cv2.rectangle(frame_resized, (ld, td), (rd, bd), color, 2)
                cv2.putText(frame_resized, name, (ld, td - 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)

                if relation == "Stranger":
                    current_frame_unidentified.append((t, r, b, l))

            new_active_unknowns = {}
            for (t, r, b, l) in current_frame_unidentified:
                center = ((t + b) / 2, (l + r) / 2)
                matched_id = None
                for tid, data in self.active_unknowns.items():
                    pt, pr, pb, pl = data["last_pos"]
                    prev_center = ((pt + pb) / 2, (pl + pr) / 2)
                    if math.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2) < 100:
                        matched_id = tid
                        break

                if matched_id is not None:
                    u_data = self.active_unknowns[matched_id]
                    u_data["last_pos"] = (t, r, b, l)
                    u_data["count"] += 1
                    if u_data["count"] >= TRIGGER_THRESHOLD and not u_data["is_saving"]:
                        u_data["is_saving"] = True
                        u_data["start_time"] = time.time()
                        u_data["buffer"] = []
                    if u_data["is_saving"]:
                        elapsed = time.time() - u_data["start_time"]
                        if elapsed < SAVE_DURATION:
                            crop = frame[max(0, t):min(frame.shape[0], b), max(0, l):min(frame.shape[1], r)]
                            if crop.size > 0:
                                u_data["buffer"].append(crop.copy())
                            new_active_unknowns[matched_id] = u_data
                        else:
                            crop = (u_data["buffer"][len(u_data["buffer"]) // 2].copy()
                                    if u_data["buffer"] else None)
                            self._add_pending_unknown(crop)
                            u_data["is_saving"] = False
                            u_data["count"] = -150
                    else:
                        new_active_unknowns[matched_id] = u_data
                else:
                    new_active_unknowns[self.next_id] = {"count": 1, "is_saving": False, "buffer": [], "start_time": 0, "last_pos": (t, r, b, l)}
                    self.next_id += 1

            self.active_unknowns = new_active_unknowns
            with self._lock:
                self.current_frame = frame_resized.copy()
                self.detected_list = detected_list[:8]
        except Exception:
            pass
        self.after(30, self._run_one_frame)

    def _poll_ui(self):
        if not self.winfo_exists() or not self.running:
            return
        with self._lock:
            cf = self.current_frame
            frame = cf.copy() if (cf is not None and hasattr(cf, "size") and cf.size > 0) else cf
            detected_list = []
            for p in self.detected_list:
                img = p.get("image")
                if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                    detected_list.append({"name": p.get("name", "?"), "rel": p.get("rel", "?"), "image": img.copy(), "notes": p.get("notes", "")})
                else:
                    detected_list.append({"name": p.get("name", "?"), "rel": p.get("rel", "?"), "image": img, "notes": p.get("notes", "")})

        # Auto-cancel pending registrations after 5 minutes
        now = time.time()
        self.pending_unknowns = [p for p in self.pending_unknowns
                                 if (now - p["created_at"]) <= PENDING_REGISTRATION_TIMEOUT_SEC]

        if frame is not None and frame.size > 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                photo = ctk.CTkImage(light_image=pil_img, size=MA_VIDEO_SIZE)
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo
                self.video_label._pil_image = pil_img
            except Exception:
                pass

        # When there is a pending registration, never rebuild the sidebar so the form stays and user can type.
        # Use a longer poll interval (250ms) when pending so we don't risk any refresh while typing.
        pending_list = list(self.pending_unknowns)
        pending_ids = tuple(p["id"] for p in pending_list)
        poll_delay = 250 if pending_list else 80
        if pending_list:
            if self._pending_ids_shown == pending_ids:
                self.after(poll_delay, self._poll_ui)
                return
            self._pending_ids_shown = pending_ids
        else:
            self._pending_ids_shown = ()
        det_sig = tuple((p.get("name", "?"), p.get("rel", "?")) for p in detected_list[:6])
        if not pending_list and self._last_sidebar_state == det_sig:
            self.after(poll_delay, self._poll_ui)
            return
        if not pending_list:
            self._last_sidebar_state = det_sig

        for w in self.sidebar_scroll.winfo_children():
            w.destroy()

        def make_unknown_form(card, img, name_entry, rel_entry, image_source, pending_item=None):
            hint = ctk.CTkLabel(
                card,
                text="Enter name and relationship in the blank fields below so we know who this is.",
                font=ctk.CTkFont(size=10),
                text_color="#aaa",
                wraplength=260,
            )
            hint.pack(anchor="w", padx=8, pady=(4, 2))
            btn_row = ctk.CTkFrame(card, fg_color="transparent")
            btn_row.pack(anchor="w", padx=8, pady=(0, 8))
            reg_btn = ctk.CTkButton(
                btn_row, text="Register", width=100, height=28, corner_radius=14,
                fg_color="white", text_color="black",
                command=lambda ne=name_entry, re=rel_entry, src=image_source, pend=pending_item: self._register_unknown_from_sidebar(ne, re, src, pend)
            )
            reg_btn.pack(side="left", padx=(0, 8))
            if pending_item is not None:
                cancel_btn = ctk.CTkButton(
                    btn_row, text="Cancel", width=80, height=28, corner_radius=14,
                    fg_color="#444", text_color="white",
                    command=lambda p=pending_item: self._cancel_pending_registration(p)
                )
                cancel_btn.pack(side="left")

        # First: captured unknowns (kept until registered, cancelled, or 5 min timeout; not removed when person leaves frame)
        for pending_item in pending_list:
            card = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1a1a1a", corner_radius=10)
            card.pack(fill="x", pady=6)
            img = pending_item.get("image")
            if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                try:
                    img = np.asarray(img).copy()
                    img_rgb = cv2.cvtColor(cv2.resize(img, (50, 50)), cv2.COLOR_BGR2RGB)
                    pil_thumb = Image.fromarray(img_rgb.copy())
                    thumb = ctk.CTkImage(light_image=pil_thumb, size=(50, 50))
                    lbl = ctk.CTkLabel(card, image=thumb, text="")
                    lbl.image = thumb
                    lbl._pil_image = pil_thumb
                    lbl.pack(side="left", padx=8, pady=8)
                except Exception:
                    pass
            ctk.CTkLabel(card, text="Unknown", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=8, pady=(8, 0))
            ctk.CTkLabel(card, text="Stranger", font=ctk.CTkFont(size=11), text_color="#888").pack(anchor="w", padx=8, pady=(0, 4))
            name_entry = ctk.CTkEntry(card, width=240, height=28, font=ctk.CTkFont(size=12), placeholder_text="Name")
            name_entry.pack(anchor="w", padx=8, pady=(2, 2))
            rel_entry = ctk.CTkEntry(card, width=240, height=28, font=ctk.CTkFont(size=12), placeholder_text="Relationship")
            rel_entry.pack(anchor="w", padx=8, pady=(0, 2))
            make_unknown_form(card, img, name_entry, rel_entry, pending_item, pending_item)

        # Then: live detected people (skip live unknowns when we have pendings to avoid duplicate register cards)
        live_list = detected_list[:6]
        if pending_list:
            live_list = [p for p in live_list if p.get("name") != "Unknown" and p.get("rel") != "Stranger"]
        for i, person in enumerate(live_list):
            card = ctk.CTkFrame(self.sidebar_scroll, fg_color="#1a1a1a", corner_radius=10)
            card.pack(fill="x", pady=6)
            img = person.get("image")
            if img is not None and isinstance(img, np.ndarray) and img.size > 0:
                try:
                    img = np.asarray(img).copy()
                    img_rgb = cv2.cvtColor(cv2.resize(img, (50, 50)), cv2.COLOR_BGR2RGB)
                    pil_thumb = Image.fromarray(img_rgb.copy())
                    thumb = ctk.CTkImage(light_image=pil_thumb, size=(50, 50))
                    lbl = ctk.CTkLabel(card, image=thumb, text="")
                    lbl.image = thumb
                    lbl._pil_image = pil_thumb
                    lbl.pack(side="left", padx=8, pady=8)
                except Exception:
                    pass
            name = person.get("name", "?")
            rel = person.get("rel", "?")
            is_unknown = (name == "Unknown" or rel == "Stranger")

            ctk.CTkLabel(card, text=name, font=ctk.CTkFont(size=17, weight="bold")).pack(anchor="w", padx=8, pady=(8, 0))
            ctk.CTkLabel(card, text=rel, font=ctk.CTkFont(size=14, weight="bold"), text_color="#aaa").pack(anchor="w", padx=8, pady=(0, 4))

            if is_unknown:
                name_entry = ctk.CTkEntry(card, width=240, height=28, font=ctk.CTkFont(size=12), placeholder_text="Name")
                name_entry.pack(anchor="w", padx=8, pady=(2, 2))
                rel_entry = ctk.CTkEntry(card, width=240, height=28, font=ctk.CTkFont(size=12), placeholder_text="Relationship")
                rel_entry.pack(anchor="w", padx=8, pady=(0, 2))
                make_unknown_form(card, img, name_entry, rel_entry, person, None)
            else:
                notes = (person.get("notes") or "").strip()
                if notes:
                    bio_short = (notes[:80] + "…") if len(notes) > 80 else notes
                    ctk.CTkLabel(card, text=bio_short, font=ctk.CTkFont(size=12, weight="bold"), text_color="#888", wraplength=260).pack(anchor="w", padx=8, pady=(0, 8))
                else:
                    ctk.CTkLabel(card, text="", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=8, pady=(0, 8))

        self.after(poll_delay, self._poll_ui)

    def _register_unknown_from_sidebar(self, name_entry, rel_entry, image_source, pending_item=None):
        """Register an unknown person using name/relation entered via keyboard. Removes pending_item if provided."""
        n = name_entry.get().strip()
        r = rel_entry.get().strip() or "Stranger"
        if not n:
            messagebox.showwarning("Register", "Please enter a name.")
            return
        img = image_source.get("image") if isinstance(image_source, dict) else image_source
        path = ""
        if img is not None and HAS_NUMPY and isinstance(img, np.ndarray) and img.size > 0:
            os.makedirs(IMAGE_FOLDER, exist_ok=True)
            path = os.path.join(IMAGE_FOLDER, f"ma_kb_{int(time.time())}.jpg")
            try:
                img = np.asarray(img, dtype=np.uint8).copy(order="C")
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                cv2.imwrite(path, img)
            except Exception:
                path = ""
        new_person = {"name": n, "relation": r, "notes": "", "image": path}
        self.app.db["people"].append(new_person)
        save_db(self.app.db)
        self._reload_known_faces()
        if pending_item is not None and pending_item in self.pending_unknowns:
            self.pending_unknowns.remove(pending_item)
        messagebox.showinfo("Registered", f"Added {n} as {r}.")

    def _cancel_pending_registration(self, pending_item):
        """Cancel the current registration; pending is removed."""
        if pending_item in self.pending_unknowns:
            self.pending_unknowns.remove(pending_item)
        self._last_sidebar_state = None


# ---------------- RUN ----------------
if __name__=="__main__":
    app=App()
    app.mainloop()
